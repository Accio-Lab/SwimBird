import torch
import logging

import numpy as np
import json
import random
# from datasets import Dataset

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_user_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "user":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples


def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples


def replace_visual_spectial_tokens(texts):

    update_texts = []
    for i, text in enumerate(texts):
        prev, after = text.split("<|im_start|>assistant")
        update_texts.append(prev + "<|im_start|>assistant" + after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>"))
        
    return update_texts


def replace_latent(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    assistant_pattern: torch.Tensor,
    pad_token: int = 0
):
    """
    Replace tokens between start_token and end_token with replacement_token,
    keeping original length, for all valid pairs after assistant pattern.
    """
    batch_size = input_ids.shape[0]
    processed_sequences = []
    pattern_len = len(assistant_pattern)

    for b in range(batch_size):
        seq = input_ids[b][attention_mask[b] == 1]
        seq_len = len(seq)
        
        # Find last occurrence of assistant pattern
        assistant_pos = -1
        for i in range(seq_len - pattern_len, -1, -1):
            if torch.equal(seq[i:i+pattern_len], assistant_pattern):
                assistant_pos = i + pattern_len - 1
                break
        
        if assistant_pos == -1:
            processed_sequences.append(seq)
            continue
        
        # Find all start/end positions
        start_mask = (seq == start_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
        end_mask = (seq == end_token) & (torch.arange(seq_len, device=seq.device) > assistant_pos)
        start_positions = start_mask.nonzero().squeeze(-1)
        end_positions = end_mask.nonzero().squeeze(-1)
        
        if len(start_positions) == 0 or len(end_positions) == 0:
            processed_sequences.append(seq)
            continue
        
        # Match each start with its first following end
        # Using searchsorted: for each start, find first end that comes after it
        valid_pairs = []
        for s_pos in start_positions:
            idx = torch.searchsorted(end_positions, s_pos, right=True)
            if idx < len(end_positions):
                valid_pairs.append((s_pos.item(), end_positions[idx].item()))
        
        if not valid_pairs:
            processed_sequences.append(seq)
            continue
        
        # Create replacement mask and new values
        new_seq = seq.clone()
        for s_pos, e_pos in valid_pairs:
            # Replace tokens between start and end (exclusive)
            new_seq[s_pos+1:e_pos] = replacement_token
        
        processed_sequences.append(new_seq)
    
    # Re-pad sequences
    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = torch.zeros((batch_size, new_max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    
    for b, seq in enumerate(processed_sequences):
        seq_len = seq.size(0)
        new_input_ids[b, :seq_len] = seq
        new_attention_mask[b, :seq_len] = 1
    
    return new_input_ids, new_attention_mask



def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:

    seq_len = row.size(0)
    pat_len = pattern.size(0)
    
    # Naive scan over all possible start positions
    for start_idx in range(seq_len - pat_len + 1):
        # Compare row[start_idx : start_idx + pat_len] to pattern
        if torch.all(row[start_idx : start_idx + pat_len] == pattern):
            return start_idx
    return -1


def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        # Mask pad tokens
        row[row == pad_token_idx] = -100
        # Mask image tokens
        row[row == img_token_idx] = -100
    
    return labels


def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        # For every position after the first <image_start_token>,
        # if the token is <image_token>, set mask = 1
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1

    return mask
