import hashlib
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


class HashingTextEncoder(nn.Module):
    """Frozen deterministic text encoder used when SentenceTransformer is unavailable."""

    def __init__(self, hidden_dim: int = 384, max_tokens: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_tokens = int(max_tokens)
        self.backend = "hashing"

    def forward(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or torch.device("cpu")
        feats = torch.zeros((len(texts), self.hidden_dim), dtype=torch.float32, device=device)
        for row, text in enumerate(texts):
            tokens = TOKEN_RE.findall(str(text).lower())[: self.max_tokens]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                idx = value % self.hidden_dim
                sign = 1.0 if ((value >> 16) & 1) else -1.0
                feats[row, idx] += sign
        return F.normalize(feats, dim=-1, eps=1e-12)


class SentenceTransformerTextEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=str(device))
        self.hidden_dim = int(self.model.get_sentence_embedding_dimension())
        self.backend = "sentence_transformer"

    def forward(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        embs = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if device is not None:
            embs = embs.to(device)
        return embs.float()


def build_text_encoder(
    backend: str,
    model_name: str,
    hidden_dim: int,
    device: torch.device,
    llm_pooling: str = "embedding_only",
) -> nn.Module:
    backend = backend.lower()
    if backend == "toolhcl":
        backend = "toolhcl_llm"
    if backend not in {"auto", "sentence_transformer", "hashing", "toolhcl_st", "toolhcl_llm"}:
        raise ValueError(
            "encoder_backend must be one of auto, sentence_transformer, hashing, toolhcl, toolhcl_st, toolhcl_llm"
        )

    if backend == "toolhcl_st":
        model_path = resolve_toolhcl_embedder_path(model_name)
        return SentenceTransformerTextEncoder(model_path, device=device)

    if backend == "toolhcl_llm":
        model_path = resolve_toolhcl_llama_path(model_name)
        return ToolHCLLlamaTextEncoder(
            model_name_or_path=model_path,
            device=device,
            pooling=llm_pooling,
        )

    if backend in {"auto", "sentence_transformer"}:
        try:
            encoder = SentenceTransformerTextEncoder(model_name, device=device)
            if backend == "auto" and encoder.hidden_dim != hidden_dim:
                print(
                    f"[model] SentenceTransformer dim={encoder.hidden_dim}; overriding hidden_dim={hidden_dim}."
                )
            return encoder
        except Exception as exc:
            if backend == "sentence_transformer":
                raise
            print(f"[model] SentenceTransformer unavailable ({exc}); using frozen hashing encoder.")

    return HashingTextEncoder(hidden_dim=hidden_dim)


def _load_toolhcl_config_attr(attr: str) -> Optional[str]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ToolHCHL"))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import config  # type: ignore

        value = getattr(config, attr, None)
        return str(value) if value is not None else None
    except Exception:
        return None


def resolve_toolhcl_embedder_path(model_name: str) -> str:
    if model_name and model_name != "auto":
        return model_name
    configured = _load_toolhcl_config_attr("EMBEDDER_PATH")
    if configured and os.path.exists(configured):
        return configured
    return "all-MiniLM-L6-v2"


def resolve_toolhcl_llama_path(model_name: str) -> str:
    if model_name and model_name != "auto":
        return model_name
    configured = _load_toolhcl_config_attr("LLAMA_PATH")
    if configured:
        return configured
    return "ToolHCHL/models_hf/Meta-Llama-3-8B"


class ToolHCLLlamaTextEncoder(nn.Module):
    """Frozen ToolHCL Llama text encoder.

    This mirrors ToolHCL's tokenization and mean-pooling input path. By default
    it uses the embedding-only path used by ToolHCL base training, which avoids
    full transformer passes while keeping the same Llama token embedding space.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        pooling: str = "embedding_only",
        embedding_layer_idx: int = 1,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if pooling not in {"embedding_only", "transformer_layer"}:
            raise ValueError("toolhcl_llm pooling must be embedding_only or transformer_layer")

        self.model_name = model_name_or_path
        self.pooling = pooling
        self.embedding_layer_idx = int(embedding_layer_idx)
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_dim = int(self.llm.config.hidden_size)
        self.backend = "toolhcl_llm"
        self.ln = nn.LayerNorm(self.hidden_dim, dtype=dtype).to(device)
        for param in self.ln.parameters():
            param.requires_grad = False

    def forward(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.llm.parameters()).device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            if self.pooling == "embedding_only":
                hidden = self.llm.model.embed_tokens(inputs.input_ids)
                hidden = self.ln(hidden)
            else:
                outputs = self.llm(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = outputs.hidden_states[self.embedding_layer_idx]
            mask = inputs.attention_mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return pooled.float()


class ToolL2PModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_tools: int,
        prompt_pool_size: int = 20,
        prompt_length: int = 5,
        top_n: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = int(hidden_dim)
        self.num_tools = int(num_tools)
        self.prompt_pool_size = int(prompt_pool_size)
        self.prompt_length = int(prompt_length)
        self.top_n = int(top_n)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_size, prompt_length, hidden_dim) * 0.02)
        self.prompt_keys = nn.Parameter(torch.randn(prompt_pool_size, hidden_dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_tools)

    def forward(
        self,
        query_text: List[str],
        *,
        prompt_sampling: bool = False,
        prompt_top_m: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.prompt_keys.device
        with torch.no_grad():
            q = self.encoder(query_text, device=device)
        q = q.to(device=device, dtype=self.prompt_keys.dtype)

        q_norm = F.normalize(q, dim=-1, eps=1e-12)
        k_norm = F.normalize(self.prompt_keys, dim=-1, eps=1e-12)
        sim = q_norm @ k_norm.t()

        top_n = min(self.top_n, self.prompt_pool_size)
        if prompt_sampling and self.training:
            top_m = int(prompt_top_m) if prompt_top_m is not None else top_n
            top_m = max(top_n, min(top_m, self.prompt_pool_size))
            top_m_ids = sim.topk(top_m, dim=-1).indices
            if top_m > top_n:
                sample_order = torch.rand((sim.shape[0], top_m), device=device).argsort(dim=-1)
                sample_pos = sample_order[:, :top_n]
                top_ids = top_m_ids.gather(1, sample_pos)
            else:
                top_ids = top_m_ids
        else:
            top_ids = sim.topk(top_n, dim=-1).indices
        selected_prompts = self.prompt_pool[top_ids]
        selected_keys = self.prompt_keys[top_ids]
        p = selected_prompts.mean(dim=(1, 2))

        h = self.fusion(torch.cat([q, p, q * p, q + p], dim=-1))
        logits = self.classifier(h)
        return {
            "logits": logits,
            "query_embedding": q,
            "selected_keys": selected_keys,
            "selected_prompt_ids": top_ids,
            "selected_prompt_indices": top_ids,
            "prompt_similarities": sim,
        }


def build_model(
    *,
    encoder_backend: str,
    encoder_model: str,
    hidden_dim: int,
    num_tools: int,
    prompt_pool_size: int,
    prompt_length: int,
    top_n: int,
    device: torch.device,
    llm_pooling: str = "embedding_only",
) -> ToolL2PModel:
    encoder = build_text_encoder(encoder_backend, encoder_model, hidden_dim, device, llm_pooling=llm_pooling)
    actual_dim = getattr(encoder, "hidden_dim", hidden_dim)
    if actual_dim != hidden_dim:
        hidden_dim = int(actual_dim)
    model = ToolL2PModel(
        encoder=encoder,
        hidden_dim=hidden_dim,
        num_tools=num_tools,
        prompt_pool_size=prompt_pool_size,
        prompt_length=prompt_length,
        top_n=top_n,
    )
    return model.to(device)


def l2p_loss(
    outputs: Dict[str, torch.Tensor],
    target_tool_ids: torch.Tensor,
    visible_ids: Iterable[int],
    lambda_key: float,
    num_tools: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    device = logits.device
    visible_tensor = torch.tensor([int(x) for x in visible_ids], dtype=torch.long, device=device)
    if visible_tensor.numel() == 0:
        raise ValueError("visible_ids is empty.")

    target_tool_ids = target_tool_ids.to(device)
    local_lookup = torch.full((num_tools,), -1, dtype=torch.long, device=device)
    local_lookup[visible_tensor] = torch.arange(visible_tensor.numel(), device=device)
    target_local = local_lookup[target_tool_ids]
    if torch.any(target_local < 0):
        missing = target_tool_ids[target_local < 0].detach().cpu().tolist()
        raise ValueError(f"Targets not present in visible_ids: {missing[:10]}")

    visible_logits = logits.index_select(1, visible_tensor)
    loss_cls = F.cross_entropy(visible_logits, target_local)

    q_norm = F.normalize(outputs["query_embedding"], dim=-1, eps=1e-12)
    selected_key_norm = F.normalize(outputs["selected_keys"], dim=-1, eps=1e-12)
    cos = (q_norm.unsqueeze(1) * selected_key_norm).sum(dim=-1)
    loss_key = 1.0 - cos.mean()
    loss = loss_cls + float(lambda_key) * loss_key
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_key": float(loss_key.detach().cpu()),
    }


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_mask.shape != logits.shape:
        raise ValueError(
            f"candidate_mask shape {tuple(candidate_mask.shape)} does not match logits {tuple(logits.shape)}"
        )
    labels = labels.to(logits.device)
    row_ids = torch.arange(labels.numel(), device=logits.device)
    candidate_mask = candidate_mask.bool().clone()
    candidate_mask[row_ids, labels] = True
    masked_logits = logits.masked_fill(~candidate_mask, -1e9)
    return F.cross_entropy(masked_logits, labels)


def _as_id_tensor(ids: Sequence[int], device: torch.device) -> torch.Tensor:
    if not ids:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor([int(x) for x in ids], dtype=torch.long, device=device)


def _sample_negatives_for_row(
    source: torch.Tensor,
    gold: int,
    k: int,
) -> torch.Tensor:
    if source.numel() == 0 or k <= 0:
        return source.new_empty(0)
    filtered = source[source != int(gold)]
    if filtered.numel() <= k:
        return filtered
    perm = torch.randperm(filtered.numel(), device=filtered.device)[:k]
    return filtered.index_select(0, perm)


def build_candidate_mask(
    *,
    target_tool_ids: torch.Tensor,
    num_tools: int,
    loss_type: str,
    visible_ids: Sequence[int],
    current_task_tool_ids: Sequence[int],
    old_tool_ids: Sequence[int],
    num_current_negatives: int,
    num_old_negatives: int,
) -> torch.Tensor:
    device = target_tool_ids.device
    batch_size = int(target_tool_ids.numel())
    mask = torch.zeros((batch_size, int(num_tools)), dtype=torch.bool, device=device)

    if loss_type == "global_ce":
        visible = _as_id_tensor(visible_ids, device)
        if visible.numel() == 0:
            raise ValueError("visible_ids is empty for global_ce.")
        mask[:, visible] = True
    elif loss_type == "local_ce":
        current = _as_id_tensor(current_task_tool_ids, device)
        if current.numel() == 0:
            raise ValueError("current_task_tool_ids is empty for local_ce.")
        mask[:, current] = True
    elif loss_type == "sampled_ce":
        current = _as_id_tensor(current_task_tool_ids, device)
        old = _as_id_tensor(old_tool_ids, device)
        for row, gold in enumerate(target_tool_ids.detach().tolist()):
            mask[row, int(gold)] = True
            cur_neg = _sample_negatives_for_row(current, int(gold), int(num_current_negatives))
            old_neg = _sample_negatives_for_row(old, int(gold), int(num_old_negatives))
            if cur_neg.numel() > 0:
                mask[row, cur_neg] = True
            if old_neg.numel() > 0:
                mask[row, old_neg] = True
    else:
        raise ValueError("loss_type must be one of global_ce, local_ce, sampled_ce.")

    row_ids = torch.arange(batch_size, device=device)
    mask[row_ids, target_tool_ids] = True
    return mask


def prompt_balance_loss(
    prompt_similarities: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    temperature = max(float(temperature), 1e-6)
    probs = torch.softmax(prompt_similarities / temperature, dim=-1)
    mean_probs = probs.mean(dim=0)
    prompt_count = mean_probs.numel()
    uniform = torch.full_like(mean_probs, 1.0 / float(prompt_count))
    eps = 1e-12
    loss = torch.sum(mean_probs * (torch.log(mean_probs + eps) - torch.log(uniform + eps)))
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps))
    max_entropy = torch.log(torch.tensor(float(prompt_count), device=prompt_similarities.device))
    normalized_entropy = entropy / torch.clamp(max_entropy, min=eps)
    return loss, {
        "prompt_soft_entropy": float(entropy.detach().cpu()),
        "prompt_soft_entropy_normalized": float(normalized_entropy.detach().cpu()),
        "prompt_soft_max_prob": float(mean_probs.max().detach().cpu()),
        "prompt_soft_min_prob": float(mean_probs.min().detach().cpu()),
    }


def l2p_regularized_loss(
    outputs: Dict[str, torch.Tensor],
    target_tool_ids: torch.Tensor,
    *,
    loss_type: str,
    visible_ids: Sequence[int],
    current_task_tool_ids: Sequence[int],
    old_tool_ids: Sequence[int],
    num_current_negatives: int,
    num_old_negatives: int,
    lambda_key: float,
    prompt_balance_loss_weight: float,
    num_tools: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    device = logits.device
    target_tool_ids = target_tool_ids.to(device)

    candidate_mask = build_candidate_mask(
        target_tool_ids=target_tool_ids,
        num_tools=num_tools,
        loss_type=loss_type,
        visible_ids=visible_ids,
        current_task_tool_ids=current_task_tool_ids,
        old_tool_ids=old_tool_ids,
        num_current_negatives=num_current_negatives,
        num_old_negatives=num_old_negatives,
    )
    loss_cls = masked_cross_entropy(logits, target_tool_ids, candidate_mask)

    q_norm = F.normalize(outputs["query_embedding"], dim=-1, eps=1e-12)
    selected_key_norm = F.normalize(outputs["selected_keys"], dim=-1, eps=1e-12)
    cos = (q_norm.unsqueeze(1) * selected_key_norm).sum(dim=-1)
    loss_key = 1.0 - cos.mean()

    balance, balance_stats = prompt_balance_loss(outputs["prompt_similarities"])
    loss = loss_cls + float(lambda_key) * loss_key
    if prompt_balance_loss_weight > 0:
        loss = loss + float(prompt_balance_loss_weight) * balance

    parts = {
        "loss": float(loss.detach().cpu()),
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_key": float(loss_key.detach().cpu()),
        "loss_balance": float(balance.detach().cpu()),
        "candidate_count_mean": float(candidate_mask.sum(dim=1).float().mean().detach().cpu()),
        **balance_stats,
    }
    return loss, parts


def checkpoint_config(
    model: ToolL2PModel,
    *,
    stage: str,
    encoder_backend: str,
    encoder_model: str,
    llm_pooling: str,
    split_path: str,
    mapping_path: str,
    l2p_variant: str = "no_replay",
    loss_type: str = "global_ce",
) -> Dict[str, object]:
    actual_backend = getattr(model.encoder, "backend", encoder_backend)
    actual_model = getattr(model.encoder, "model_name", encoder_model)
    return {
        "stage": stage,
        "encoder_backend": actual_backend,
        "encoder_model": actual_model,
        "llm_pooling": llm_pooling,
        "hidden_dim": model.hidden_dim,
        "num_tools": model.num_tools,
        "prompt_pool_size": model.prompt_pool_size,
        "prompt_length": model.prompt_length,
        "top_n": model.top_n,
        "split_path": split_path,
        "mapping_path": mapping_path,
        "l2p_variant": l2p_variant,
        "loss_type": loss_type,
    }


def save_checkpoint(path: str, model: ToolL2PModel, config: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
    trainable_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    payload = {
        "model_state": trainable_state,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint_model(
    path: str,
    device: torch.device,
    encoder_backend_override: Optional[str] = None,
) -> Tuple[ToolL2PModel, Dict[str, object], Dict[str, object]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))
    if not config:
        raise ValueError(f"Checkpoint {path} does not contain a config.")
    backend = encoder_backend_override or str(config.get("encoder_backend", "auto"))
    model = build_model(
        encoder_backend=backend,
        encoder_model=str(config.get("encoder_model", "all-MiniLM-L6-v2")),
        hidden_dim=int(config["hidden_dim"]),
        num_tools=int(config["num_tools"]),
        prompt_pool_size=int(config.get("prompt_pool_size", 20)),
        prompt_length=int(config.get("prompt_length", 5)),
        top_n=int(config.get("top_n", 5)),
        device=device,
        llm_pooling=str(config.get("llm_pooling", "embedding_only")),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    return model, config, ckpt
