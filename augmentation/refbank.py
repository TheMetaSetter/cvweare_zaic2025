from pathlib import Path
import re

def build_augmented_ref_bank(root):
    root = Path(root)
    bank = {}
    for cls_dir in root.iterdir():
        if cls_dir.is_dir():
            bank[cls_dir.name] = list(cls_dir.glob("*.png"))
    return bank

def sample_folder_to_ref_key(sample_folder: str, ref_bank: dict) -> str | None:
    name = Path(sample_folder).name          # "Laptop_0" -> "Laptop_0"
    # remove trailing "_123" or trailing digits, spaces, dashes
    base = re.sub(r'[_\-\s]*\d+$', '', name).lower()  # -> "laptop"
    if base in ref_bank:
        return base
    # fallback: try fuzzy startswith/contains matches
    for k in ref_bank:
        kl = k.lower()
        if kl.startswith(base) or base.startswith(kl) or base in kl:
            return k
    return None  # no match: skip or warn