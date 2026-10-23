"""Libas Lab - a single-file Pakistani wardrobe stylist and trend radar.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import math
import random
import textwrap
from dataclasses import dataclass
from datetime import date
from html import escape
from itertools import product
from typing import Sequence
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Vocabulary and styling configuration
# --------------------------------------------------------------------------

GENDER_OPTIONS = ("Women", "Men")
SEASON_OPTIONS = ("Summer", "Winter")
FORMALITY_LABELS = ("Very casual", "Smart casual", "Festive", "Formal")


@dataclass(frozen=True)
class Occasion:
    key: str
    label: str
    target_formality: int
    description: str
    pinterest_context: str


OCCASIONS = (
    Occasion("casual", "Casual day", 0, "Errands, university, chai runs", "casual street style"),
    Occasion("smart", "Smart casual", 1, "Office, presentations, brunch", "smart casual"),
    Occasion("festive", "Eid & festive", 2, "Eid, mehndi, family functions", "eid festive"),
    Occasion("formal", "Wedding & evening", 3, "Nikkah, walima, dinners", "wedding formal"),
)
OCCASION_BY_KEY = {occasion.key: occasion for occasion in OCCASIONS}


@dataclass(frozen=True)
class GarmentCandidate:
    label: str
    prompt: str
    style: str
    formality: int


TOP_CANDIDATES: dict[str, tuple[GarmentCandidate, ...]] = {
    "Women": (
        GarmentCandidate("Lawn kurti", "a photo of a woman wearing a printed lawn kurti in Pakistan", "Eastern", 1),
        GarmentCandidate("Embroidered kurta", "a photo of a woman wearing an embroidered Pakistani kurta", "Eastern", 2),
        GarmentCandidate("Festive kurta set", "a photo of a woman wearing a formal Pakistani kurta set for Eid", "Eastern", 3),
        GarmentCandidate("Blouse", "a photo of a woman wearing a western blouse", "Western", 1),
        GarmentCandidate("T-shirt", "a photo of a woman wearing a casual t-shirt", "Western", 0),
        GarmentCandidate("Formal shirt", "a photo of a woman wearing a formal button-up shirt", "Western", 3),
        GarmentCandidate("Co-ord top", "a photo of a woman wearing a matching co-ord set top", "Western", 1),
        GarmentCandidate("Hoodie or sweatshirt", "a photo of a woman wearing a hoodie or sweatshirt", "Western", 0),
    ),
    "Men": (
        GarmentCandidate("Kurta", "a photo of a man wearing a plain Pakistani kurta", "Eastern", 1),
        GarmentCandidate("Embroidered kurta", "a photo of a man wearing an embroidered festive kurta", "Eastern", 2),
        GarmentCandidate("Casual button-up shirt", "a photo of a man wearing a casual button-up shirt", "Western", 1),
        GarmentCandidate("Formal dress shirt", "a photo of a man wearing a formal dress shirt", "Western", 3),
        GarmentCandidate("T-shirt", "a photo of a man wearing a t-shirt", "Western", 0),
        GarmentCandidate("Polo shirt", "a photo of a man wearing a polo shirt", "Western", 1),
        GarmentCandidate("Hoodie or sweatshirt", "a photo of a man wearing a hoodie or sweatshirt", "Western", 0),
        GarmentCandidate("Sweater", "a photo of a man wearing a knitted sweater", "Western", 1),
    ),
}

BOTTOM_CANDIDATES: dict[str, tuple[GarmentCandidate, ...]] = {
    "Women": (
        GarmentCandidate("Shalwar", "a photo of a woman wearing a Pakistani shalwar", "Eastern", 1),
        GarmentCandidate("Jeans", "a photo of a woman wearing blue jeans", "Western", 0),
        GarmentCandidate("Wide-leg trousers", "a photo of a woman wearing wide-leg trousers", "Western", 1),
        GarmentCandidate("Palazzo pants", "a photo of a woman wearing palazzo pants", "Eastern", 1),
        GarmentCandidate("Co-ord trousers", "a photo of a woman wearing co-ord set trousers", "Western", 1),
        GarmentCandidate("Leggings", "a photo of a woman wearing leggings", "Western", 0),
        GarmentCandidate("Skirt", "a photo of a woman wearing a skirt", "Western", 1),
        GarmentCandidate("Formal trousers", "a photo of a woman wearing tailored formal trousers", "Western", 3),
    ),
    "Men": (
        GarmentCandidate("Shalwar", "a photo of a man wearing a Pakistani shalwar", "Eastern", 1),
        GarmentCandidate("Jeans", "a photo of a man wearing blue jeans", "Western", 0),
        GarmentCandidate("Chinos", "a photo of a man wearing chino trousers", "Western", 1),
        GarmentCandidate("Cargo pants", "a photo of a man wearing cargo pants", "Western", 0),
        GarmentCandidate("Formal trousers", "a photo of a man wearing tailored formal trousers", "Western", 3),
        GarmentCandidate("Joggers", "a photo of a man wearing jogger pants", "Western", 0),
        GarmentCandidate("Shorts", "a photo of a man wearing shorts", "Western", 0),
    ),
}

PATTERN_CANDIDATES: tuple[tuple[str, str, float], ...] = (
    ("Solid", "a photo of solid-colour plain clothing fabric", 0.0),
    ("Embroidered", "a photo of embroidered traditional clothing", 0.85),
    ("Floral print", "a photo of floral printed clothing", 0.8),
    ("Checked / geometric", "a photo of checked or geometric patterned clothing", 0.65),
    ("Striped", "a photo of striped clothing", 0.55),
    ("Graphic print", "a photo of a t-shirt with a graphic print", 0.75),
    ("Textured", "a photo of textured, knitted, or self-patterned fabric", 0.45),
)

FORMALITY_CANDIDATES: tuple[tuple[str, str, int], ...] = (
    ("Very casual", "a photo of very casual everyday clothing", 0),
    ("Smart casual", "a photo of smart casual clothing", 1),
    ("Festive", "a photo of festive traditional Pakistani clothing", 2),
    ("Formal", "a photo of formal wedding or evening clothing", 3),
)

NEUTRAL_COLORS = {
    "Black", "White", "Grey", "Charcoal", "Cream", "Beige",
    "Off-white", "Tan", "Brown", "Navy",
}

COLOR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("White", "white"), ("Cream", "cream"), ("Beige", "beige"),
    ("Grey", "grey"), ("Black", "black"), ("Brown", "brown"),
    ("Navy", "navy blue"), ("Blue", "blue"), ("Light blue", "light blue"),
    ("Teal", "teal"), ("Green", "green"), ("Olive", "olive green"),
    ("Mustard", "mustard yellow"), ("Orange", "orange"), ("Red", "red"),
    ("Maroon", "maroon"), ("Pink", "pink"), ("Purple", "purple"),
)

COLOR_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a person wearing {colour} clothing",
    "a photo of a {colour} garment",
)

COLOR_PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("Black", (24, 24, 24)), ("Charcoal", (66, 66, 66)),
    ("Grey", (140, 140, 140)), ("White", (245, 244, 240)),
    ("Cream", (241, 232, 210)), ("Beige", (214, 194, 162)),
    ("Tan", (182, 146, 102)), ("Brown", (118, 80, 54)),
    ("Navy", (34, 50, 82)), ("Blue", (58, 98, 170)),
    ("Light blue", (158, 199, 222)), ("Powder blue", (186, 208, 226)),
    ("Teal", (37, 124, 126)), ("Green", (48, 116, 74)),
    ("Emerald", (28, 92, 65)), ("Olive", (108, 112, 62)),
    ("Mustard", (198, 160, 54)), ("Yellow", (232, 205, 92)),
    ("Orange", (221, 118, 49)), ("Red", (177, 43, 48)),
    ("Maroon", (107, 38, 52)), ("Pink", (224, 139, 164)),
    ("Blush", (234, 196, 195)), ("Purple", (108, 69, 137)),
)

# --------------------------------------------------------------------------
# Pakistani accessory catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Accessory:
    name: str
    group: str
    gender: str
    occasions: tuple[str, ...]
    styles: tuple[str, ...]
    formality: tuple[int, int]
    colors: tuple[str, ...]
    seasons: tuple[str, ...]
    why: str
    query: str


BOTH_SEASONS = ("Summer", "Winter")
ALL_OCCASIONS = ("casual", "smart", "festive", "formal")
GROUPS = ("Footwear", "Finishing touch", "Carry / layer")

ACCESSORIES: tuple[Accessory, ...] = (
    Accessory("Peshawari chappal", "Footwear", "Men", ("casual", "smart", "festive"), ("Eastern",), (0, 2), ("Brown", "Tan", "Black"), ("Summer", "Winter"), "Leather chappals keep a kurta look grounded and are normal street wear in Pakistan.", "men peshawari chappal kurta outfit pakistan"),
    Accessory("White leather sneakers", "Footwear", "Men", ("casual", "smart"), ("Western", "Eastern"), (0, 1), ("White", "Cream"), BOTH_SEASONS, "Clean sneakers modernise shalwar or chinos without making the outfit look careless.", "men white sneakers shalwar kameez outfit pakistan"),
    Accessory("Dark leather loafers", "Footwear", "Men", ("smart", "formal"), ("Western",), (1, 3), ("Black", "Brown"), BOTH_SEASONS, "A slim loafer dresses up tailored trousers for dinners, nikkahs and office days.", "men loafers formal pakistani outfit"),
    Accessory("Embroidered khussa", "Footwear", "Men", ("festive", "formal"), ("Eastern",), (2, 3), ("Cream", "Gold", "Brown"), BOTH_SEASONS, "Khussa is the easiest way to make a plain kurta read as festive.", "men khussa with kurta pakistan"),
    Accessory("Minimal steel watch", "Finishing touch", "Men", ALL_OCCASIONS, ("Western", "Eastern"), (1, 3), ("Silver", "Black"), BOTH_SEASONS, "A small, clean watch adds polish and works with everything from polos to sherwanis.", "men minimal watch pakistani style"),
    Accessory("Silver cuff bracelet", "Finishing touch", "Men", ("casual", "festive"), ("Eastern", "Western"), (0, 2), ("Silver",), BOTH_SEASONS, "A single cuff adds personality to a plain kurta or tee without becoming loud.", "men silver cuff bracelet kurta pakistan"),
    Accessory("Classic sunglasses", "Finishing touch", "Men", ("casual",), ("Western",), (0, 1), ("Black", "Brown"), ("Summer",), "Keeps a hot-day street look sharp and is practical for Pakistani summers.", "men sunglasses casual outfit pakistan"),
    Accessory("Pocket square", "Finishing touch", "Men", ("formal",), ("Western",), (3, 3), ("White", "Cream", "Navy"), BOTH_SEASONS, "A crisp pocket square makes a blazer or waistcoat feel deliberately styled.", "men pocket square formal outfit pakistan"),
    Accessory("Leather satchel", "Carry / layer", "Men", ("smart",), ("Western",), (1, 2), ("Brown", "Black"), BOTH_SEASONS, "A structured bag completes office and presentation looks without adding bulk.", "men leather satchel office outfit pakistan"),
    Accessory("Canvas tote", "Carry / layer", "Men", ("casual",), ("Western", "Eastern"), (0, 1), ("Beige", "Olive", "Black"), BOTH_SEASONS, "A relaxed tote suits campus and errand days and tones down a dressy kurta.", "men canvas tote casual outfit pakistan"),
    Accessory("Textured waistcoat", "Carry / layer", "Men", ("festive", "formal"), ("Eastern", "Western"), (2, 3), ("Navy", "Black", "Beige"), ("Summer", "Winter"), "A waistcoat adds structure over a kurta and is a staple for mehndi and walima events.", "men waistcoat with kurta pakistan wedding"),
    Accessory("Kashmiri shawl", "Carry / layer", "Men", ("festive", "formal"), ("Eastern",), (2, 3), ("Maroon", "Cream", "Green"), ("Winter",), "A fine wool shawl layers warmth over a kurta while keeping the look traditional.", "men kashmiri shawl kurta pakistan winter"),
    Accessory("Khussa with juttis", "Footwear", "Women", ("festive", "formal"), ("Eastern",), (2, 3), ("Gold", "Cream", "Maroon"), BOTH_SEASONS, "Khussa pairs naturally with kurta sets and is comfortable through long mehndi functions.", "women khussa juttis kurta outfit pakistan"),
    Accessory("Block heels", "Footwear", "Women", ("smart", "formal"), ("Western",), (1, 3), ("Beige", "Black", "Nude"), BOTH_SEASONS, "Block heels give height on uneven lawns and work with wide-leg trousers or co-ords.", "women block heels formal outfit pakistan"),
    Accessory("White leather sneakers", "Footwear", "Women", ("casual", "smart"), ("Western", "Eastern"), (0, 1), ("White", "Cream"), BOTH_SEASONS, "Sneakers keep a kurti-and-jeans day comfortable and current.", "women white sneakers kurti outfit pakistan"),
    Accessory("Flat leather sandals", "Footwear", "Women", ("casual",), ("Eastern", "Western"), (0, 1), ("Tan", "Brown", "Gold"), ("Summer",), "Flat sandals suit summer lawn outfits and long walking days.", "women flat sandals lawn outfit pakistan summer"),
    Accessory("Jhumka earrings", "Finishing touch", "Women", ("festive", "formal"), ("Eastern",), (2, 3), ("Gold", "Silver"), BOTH_SEASONS, "Jhumkas frame the face and finish an Eastern outfit without needing a heavy necklace.", "women jhumka earrings eid outfit pakistan"),
    Accessory("Minimal gold-tone studs", "Finishing touch", "Women", ("smart", "formal"), ("Western", "Eastern"), (1, 3), ("Gold",), BOTH_SEASONS, "Small studs keep office and dinner looks refined and work with printed outfits.", "women minimal gold earrings office outfit pakistan"),
    Accessory("Statement ring", "Finishing touch", "Women", ("casual", "festive"), ("Eastern", "Western"), (1, 2), ("Silver", "Gold"), BOTH_SEASONS, "A single statement ring adds a considered detail to otherwise simple clothes.", "women statement ring casual pakistani outfit"),
    Accessory("Sunglasses", "Finishing touch", "Women", ("casual",), ("Western",), (0, 1), ("Black", "Brown"), ("Summer",), "A practical finishing touch for bright daytime looks.", "women sunglasses casual outfit pakistan"),
    Accessory("Organza dupatta", "Carry / layer", "Women", ("festive", "formal"), ("Eastern",), (2, 3), ("Cream", "Pink", "Teal"), BOTH_SEASONS, "A light organza dupatta makes a simple kurta set photograph like a festive outfit.", "women organza dupatta styling pakistan"),
    Accessory("Canvas sling bag", "Carry / layer", "Women", ("casual",), ("Western", "Eastern"), (0, 1), ("Beige", "Black", "Olive"), ("Summer", "Winter"), "A cross-body sling keeps a casual kurti day hands-free and pulls the look together.", "women casual sling bag kurti outfit pakistan"),
    Accessory("Structured tote", "Carry / layer", "Women", ("smart",), ("Western",), (1, 2), ("Beige", "Black", "Tan"), BOTH_SEASONS, "A clean tote completes workwear and holds its shape all day.", "women structured tote office outfit pakistan"),
    Accessory("Embroidered clutch", "Carry / layer", "Women", ("festive", "formal"), ("Eastern", "Western"), (2, 3), ("Gold", "Maroon", "Black"), BOTH_SEASONS, "A clutch keeps formal looks streamlined while adding a festive focal point.", "women embroidered clutch wedding outfit pakistan"),
    Accessory("Embroidered shawl", "Carry / layer", "Women", ("festive", "formal"), ("Eastern",), (2, 3), ("Cream", "Maroon"), ("Winter",), "A shawl makes a winter wedding outfit practical without hiding the kurta.", "women embroidered shawl winter wedding pakistan"),
    Accessory("Silk scarf", "Carry / layer", "Women", ("smart", "festive"), ("Western", "Eastern"), (1, 2), ("Beige", "Blush"), ("Winter",), "A small scarf adds colour near the face and warmth over a plain shirt or kurta.", "women silk scarf smart casual pakistan"),
)

# --------------------------------------------------------------------------
# FashionCLIP vision
# --------------------------------------------------------------------------

MODEL_ID = "patrickjohncyh/fashion-clip"


class ModelLoadError(RuntimeError):
    pass


def prepare_image(image: Image.Image, max_side: int = 768) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def image_to_jpeg(image: Image.Image, max_side: int = 900, quality: int = 88) -> bytes:
    prepared = prepare_image(image.copy(), max_side=max_side)
    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def name_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = (channel / 255 for channel in rgb)
    _, saturation, value = colorsys.rgb_to_hsv(r, g, b)

    if value < 0.15:
        return "Black"
    if saturation < 0.11:
        if value > 0.88:
            return "White"
        if value > 0.5:
            return "Grey"
        return "Charcoal"

    best_name = "Grey"
    best_distance = float("inf")
    for name, palette_rgb in COLOR_PALETTE:
        distance = sum((left - right) ** 2 for left, right in zip(rgb, palette_rgb))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name


def extract_palette(image: Image.Image, top_n: int = 3) -> list[dict]:
    prepared = prepare_image(image, max_side=640)
    width, height = prepared.size
    margin_x, margin_y = int(width * 0.1), int(height * 0.1)
    if width - 2 * margin_x > 8 and height - 2 * margin_y > 8:
        prepared = prepared.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    prepared = prepared.resize((160, 160), Image.Resampling.LANCZOS)

    quantized = prepared.quantize(colors=10, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=256) or []
    total = sum(count for count, _ in counts) or 1

    merged: dict[str, dict] = {}
    for count, index in counts:
        offset = index * 3
        if offset + 2 >= len(palette):
            continue
        rgb = tuple(int(channel) for channel in palette[offset:offset + 3])
        _, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
        name = name_color(rgb)
        share = count / total
        weight = share * (0.4 + 0.8 * saturation) * (0.85 + 0.3 * (1 - abs(value - 0.55)))
        if saturation < 0.09 and value > 0.94:
            weight *= 0.18
        entry = merged.setdefault(name, {"name": name, "rgb_sum": np.zeros(3), "weight": 0.0, "share": 0.0})
        entry["rgb_sum"] += np.array(rgb, dtype=float) * weight
        entry["weight"] += weight
        entry["share"] += share

    ranked = sorted(merged.values(), key=lambda item: item["weight"], reverse=True)[:top_n]
    results: list[dict] = []
    for entry in ranked:
        if entry["weight"] <= 0:
            continue
        rgb = tuple(int(round(channel)) for channel in entry["rgb_sum"] / entry["weight"])
        hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
        results.append({
            "name": entry["name"], "rgb": rgb, "hex": hex_color(rgb),
            "hsv": (round(hue, 4), round(saturation, 4), round(value, 4)),
            "share": round(entry["share"] * 100),
        })

    if not results:
        average = prepared.resize((1, 1)).getpixel((0, 0))
        hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in average))
        results = [{
            "name": name_color(average), "rgb": average, "hex": hex_color(average),
            "hsv": (round(hue, 4), round(saturation, 4), round(value, 4)), "share": 100,
        }]
    return results


def _softmax(similarities: np.ndarray) -> np.ndarray:
    shifted = similarities - similarities.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


class FashionEncoder:
    def __init__(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ModelLoadError(
                "PyTorch and Transformers are required. Run: pip install -r requirements.txt"
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CLIPModel.from_pretrained(MODEL_ID)
            self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        except Exception as exc:
            raise ModelLoadError(
                "FashionCLIP could not be loaded. Check your internet connection "
                "the first time you run the app, then try again."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        try:
            self.logit_scale = float(self.model.logit_scale.exp().item())
        except Exception:
            self.logit_scale = 100.0
        self.logit_scale = float(np.clip(self.logit_scale, 20.0, 200.0))
        self._text_cache: dict[tuple[str, ...], np.ndarray] = {}

    @property
    def device_label(self) -> str:
        return "GPU" if self.device == "cuda" else "CPU"

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 4) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [prepare_image(image) for image in images[start:start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            batches.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        key = tuple(prompts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        if not prompts:
            return np.empty((0, 512), dtype=np.float32)
        inputs = self.processor(
            text=list(prompts), return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        vectors = features.detach().cpu().numpy().astype(np.float32)
        if len(self._text_cache) > 64:
            self._text_cache.clear()
        self._text_cache[key] = vectors
        return vectors

    def zero_shot(self, embedding: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
        text_vectors = self.embed_texts(prompts)
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
        similarities = (text_vectors @ embedding) * self.logit_scale
        return _softmax(similarities)


def classify_attributes(embedding: np.ndarray, slot: str, gender: str, encoder: FashionEncoder) -> dict:
    candidates = (TOP_CANDIDATES if slot == "top" else BOTTOM_CANDIDATES)[gender]
    type_probs = encoder.zero_shot(embedding, [candidate.prompt for candidate in candidates])
    best_index = int(np.argmax(type_probs))
    best_candidate = candidates[best_index]

    pattern_probs = encoder.zero_shot(embedding, [prompt for _, prompt, _ in PATTERN_CANDIDATES])
    pattern_index = int(np.argmax(pattern_probs))
    pattern_name = PATTERN_CANDIDATES[pattern_index][0]
    pattern_strength = float(sum(
        probability * strength
        for probability, (_, _, strength) in zip(pattern_probs, PATTERN_CANDIDATES)
    ))

    formality_probs = encoder.zero_shot(embedding, [prompt for _, prompt, _ in FORMALITY_CANDIDATES])
    clip_formality = float(sum(
        probability * level
        for probability, (_, _, level) in zip(formality_probs, FORMALITY_CANDIDATES)
    ))
    blended_formality = 0.7 * best_candidate.formality + 0.3 * clip_formality
    formality = int(np.clip(round(blended_formality), 0, 3))

    return {
        "type": best_candidate.label,
        "type_confidence": round(float(type_probs[best_index]) * 100),
        "style": best_candidate.style,
        "pattern": pattern_name,
        "pattern_strength": round(pattern_strength, 3),
        "formality": formality,
        "formality_label": FORMALITY_LABELS[formality],
    }


def classify_color(embedding: np.ndarray, encoder: FashionEncoder) -> dict:
    probabilities = np.zeros(len(COLOR_CANDIDATES), dtype=float)
    for template in COLOR_PROMPT_TEMPLATES:
        prompts = [template.format(colour=phrase) for _, phrase in COLOR_CANDIDATES]
        probabilities += encoder.zero_shot(embedding, prompts)
    probabilities /= len(COLOR_PROMPT_TEMPLATES)

    best_index = int(np.argmax(probabilities))
    name = COLOR_CANDIDATES[best_index][0]
    representative = dict(COLOR_PALETTE).get(name, (180, 180, 180))
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in representative))
    return {
        "color_name": name,
        "color_hex": hex_color(representative),
        "color_rgb": representative,
        "hsv": (round(hue, 4), round(saturation, 4), round(value, 4)),
        "color_confidence": round(float(probabilities[best_index]) * 100),
    }


def analyze_garments(
    images: Sequence[Image.Image], slot: str, gender: str, encoder: FashionEncoder
) -> list[tuple[dict, np.ndarray]]:
    prepared = [prepare_image(image) for image in images]
    embeddings = encoder.embed_images(prepared)
    results: list[tuple[dict, np.ndarray]] = []
    for image, embedding in zip(prepared, embeddings):
        attributes = classify_attributes(embedding, slot, gender, encoder)
        attributes.update(classify_color(embedding, encoder))
        attributes.update({"slot": slot, "palette": extract_palette(image)})
        results.append((attributes, embedding))
    return results

# --------------------------------------------------------------------------
# Outfit scoring, accessories, and Pinterest links
# --------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_neutral(item: dict) -> bool:
    return item["color_name"] in NEUTRAL_COLORS or item["hsv"][1] < 0.22


def color_harmony(top: dict, bottom: dict) -> tuple[float, str]:
    if _is_neutral(top) and _is_neutral(bottom):
        return 0.94, f"{top['color_name']} and {bottom['color_name']} are a quiet neutral pairing that is hard to get wrong."
    if _is_neutral(top) or _is_neutral(bottom):
        anchor = top if _is_neutral(top) else bottom
        accent = bottom if anchor is top else top
        return 0.91, f"{anchor['color_name']} grounds the {accent['color_name'].lower()} piece, so the eye has somewhere to rest."

    hue_a, saturation_a = top["hsv"][0], top["hsv"][1]
    hue_b, saturation_b = bottom["hsv"][0], bottom["hsv"][1]
    delta = min(abs(hue_a - hue_b), 1 - abs(hue_a - hue_b))

    if delta < 0.045:
        return 0.87, "Tone-on-tone colours keep the outfit long and lean."
    if delta < 0.12:
        return 0.82, "These neighbouring colours blend rather than compete."
    if delta > 0.40:
        return 0.84, "The contrast reads as deliberate, especially with one piece kept simple."
    if min(saturation_a, saturation_b) > 0.5:
        return 0.58, "Two strong hues compete here; keep accessories neutral so it does not get loud."
    return 0.72, "The muted contrast works, but add a neutral shoe or bag to settle it."


def formality_fit(top: dict, bottom: dict, target: int) -> tuple[float, str, float]:
    average = (top["formality"] + bottom["formality"]) / 2
    spread = abs(top["formality"] - bottom["formality"])
    score = 1 - 0.2 * abs(average - target) - 0.08 * max(0, spread - 1)
    score = _clamp(score, 0.28, 1.0)

    difference = average - target
    if abs(difference) <= 0.5:
        message = "The formality of both pieces matches the occasion."
    elif difference > 0:
        message = "This is slightly dressier than the occasion; keep shoes relaxed."
    else:
        message = "This sits more casual than the occasion; finish it with polished footwear."
    return score, message, average


def pattern_balance(top: dict, bottom: dict) -> tuple[float, str]:
    strength_top = top["pattern_strength"]
    strength_bottom = bottom["pattern_strength"]
    if strength_top > 0.5 and strength_bottom > 0.5:
        return 0.55, "Both pieces carry strong pattern, so keep accessories very quiet."
    if max(strength_top, strength_bottom) > 0.5:
        return 0.9, "One patterned piece and one plain piece is the safest balance."
    return 0.92, "The texture mix adds depth without shouting."


def style_balance(top: dict, bottom: dict, target: int) -> tuple[float, str]:
    if top["style"] == bottom["style"]:
        return 1.0, ""
    penalty = 0.7 if target >= 2 else 0.8
    return penalty, "The Eastern and Western mix works as fusion; keep the accessories on one side of the line."


def _accessory_score(
    accessory: Accessory, top: dict, bottom: dict, gender: str,
    occasion_key: str, season: str, formality_average: float,
) -> float | None:
    if accessory.gender not in (gender, "Unisex"):
        return None
    if occasion_key not in accessory.occasions:
        return None

    score = 0.4
    low, high = accessory.formality
    distance = max(low - formality_average, 0, formality_average - high)
    score += 0.25 * _clamp(1 - 0.3 * distance, 0.3, 1.0)

    styles = {top["style"], bottom["style"]}
    if styles & set(accessory.styles):
        score += 0.2
    else:
        score -= 0.08

    if season in accessory.seasons:
        score += 0.1
    else:
        score -= 0.12

    outfit_colors = {top["color_name"], bottom["color_name"]}
    if outfit_colors & set(accessory.colors):
        score += 0.06
    elif set(accessory.colors) & NEUTRAL_COLORS:
        score += 0.04

    return _clamp(score, 0.0, 1.0)


def pinterest_url(query: str) -> str:
    return "https://www.pinterest.com/search/pins/?q=" + quote_plus(query)


def suggest_accessories(
    top: dict, bottom: dict, gender: str, occasion_key: str,
    season: str, formality_average: float | None = None,
) -> list[dict]:
    if formality_average is None:
        formality_average = (top["formality"] + bottom["formality"]) / 2

    by_group: dict[str, list[tuple[float, Accessory]]] = {group: [] for group in GROUPS}
    for accessory in ACCESSORIES:
        score = _accessory_score(
            accessory, top, bottom, gender, occasion_key, season, formality_average
        )
        if score is not None:
            by_group[accessory.group].append((score, accessory))

    suggestions: list[dict] = []
    for group in GROUPS:
        options = by_group[group]
        if not options:
            continue
        best_score, best = max(options, key=lambda option: option[0])
        suggestions.append({
            "name": best.name,
            "group": best.group,
            "why": best.why,
            "colors": best.colors,
            "match": round(best_score * 100),
            "pinterest_url": pinterest_url(best.query),
        })
    return suggestions


def outfit_pinterest_query(top: dict, bottom: dict, gender: str, occasion: Occasion, season: str) -> str:
    return " ".join(
        part for part in (
            gender.lower(), occasion.pinterest_context, "outfit pakistan",
            top["color_name"], top["type"], bottom["color_name"],
            bottom["type"], season.lower(),
        ) if part
    )


def score_outfit(top: dict, bottom: dict, occasion_key: str, gender: str, season: str) -> dict:
    occasion = OCCASION_BY_KEY[occasion_key]
    color_score, color_note = color_harmony(top, bottom)
    formality_score, formality_note, formality_average = formality_fit(
        top, bottom, occasion.target_formality
    )
    pattern_score, pattern_note = pattern_balance(top, bottom)
    style_score, style_note = style_balance(top, bottom, occasion.target_formality)

    raw_score = (
        0.38 * color_score + 0.32 * formality_score
        + 0.15 * pattern_score + 0.15 * style_score
    )
    reasons = [color_note, formality_note, pattern_note]
    if style_note:
        reasons.append(style_note)

    accessories = suggest_accessories(
        top, bottom, gender, occasion_key, season, formality_average
    )
    return {
        "top": top,
        "bottom": bottom,
        "score": round(raw_score * 100),
        "reasons": reasons,
        "accessories": accessories,
        "pinterest_url": pinterest_url(
            outfit_pinterest_query(top, bottom, gender, occasion, season)
        ),
    }


def build_outfits(
    tops: list[dict], bottoms: list[dict], gender: str,
    occasion_key: str, season: str, limit: int | None = 6,
) -> tuple[list[dict], int]:
    looks = [
        score_outfit(top, bottom, occasion_key, gender, season)
        for top, bottom in product(tops, bottoms)
    ]
    looks.sort(key=lambda look: look["score"], reverse=True)
    return (looks[:limit] if limit is not None else looks), len(looks)

# --------------------------------------------------------------------------
# Trend data and seasonal forecast
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = ("date", "keyword", "gender", "category", "interest")
AUDIENCES = ("Everyone", "Women", "Men")
HORIZON_MONTHS = 6


class TrendDataError(ValueError):
    pass


def _normalise_gender(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"men", "man", "male", "m"}:
        return "Men"
    if text in {"women", "woman", "female", "w", "f"}:
        return "Women"
    return "Unisex"


def _prepare_trend_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TrendDataError(
            "The CSV is missing these columns: " + ", ".join(missing) + ". "
            "Expected: date, keyword, gender, category, interest."
        )

    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["interest"] = pd.to_numeric(frame["interest"], errors="coerce")
    frame = frame.dropna(subset=["date", "interest", "keyword"])
    frame["keyword"] = frame["keyword"].astype(str).str.strip()
    frame = frame[frame["keyword"] != ""]
    frame["gender"] = frame["gender"].map(_normalise_gender)
    frame["category"] = frame["category"].fillna("Other").astype(str).str.strip()
    frame["category"] = frame["category"].replace("", "Other")
    frame["interest"] = frame["interest"].clip(0, 100)

    if frame.empty:
        raise TrendDataError("The CSV has no usable rows after cleaning.")

    return (
        frame.groupby(["date", "keyword", "gender", "category"], as_index=False)["interest"]
        .mean()
        .sort_values("date")
    )


def load_trend_data(source) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source)
    except Exception as exc:
        raise TrendDataError(f"Could not read the CSV: {exc}") from exc
    return _prepare_trend_frame(frame)


@dataclass(frozen=True)
class TrendSeries:
    keyword: str
    gender: str
    category: str
    base: float
    slope: float
    amplitude: float
    peak_month: int
    noise: float


TREND_SERIES: tuple[TrendSeries, ...] = (
    TrendSeries("Lawn suit designs", "Women", "Eastern", 56, 0.12, 14, 4, 2.4),
    TrendSeries("Co-ord sets", "Women", "Western", 18, 0.45, 5, 5, 2.0),
    TrendSeries("Wide-leg trousers", "Women", "Western", 22, 0.35, 4, 7, 2.0),
    TrendSeries("Pastel kurtis", "Women", "Eastern", 22, 0.25, 6, 3, 2.2),
    TrendSeries("Organza dupattas", "Women", "Eastern", 25, 0.30, 9, 11, 2.3),
    TrendSeries("Saree draping", "Women", "Eastern", 34, 0.08, 6, 11, 2.2),
    TrendSeries("Block heels", "Women", "Footwear", 22, 0.16, 5, 11, 1.8),
    TrendSeries("Skinny jeans", "Women", "Western", 52, -0.32, 4, 1, 2.0),
    TrendSeries("Shalwar kameez", "Men", "Eastern", 56, 0.05, 7, 5, 2.0),
    TrendSeries("Linen kurtas", "Men", "Eastern", 14, 0.40, 8, 5, 2.4),
    TrendSeries("Kurta pajama", "Men", "Eastern", 46, 0.03, 8, 4, 2.1),
    TrendSeries("Cargo pants", "Men", "Western", 38, 0.18, 4, 1, 2.0),
    TrendSeries("Oversized t-shirts", "Men", "Western", 35, 0.25, 5, 6, 2.0),
    TrendSeries("Waistcoat styling", "Men", "Eastern", 18, 0.15, 8, 11, 2.2),
    TrendSeries("Peshawari chappals", "Men", "Footwear", 25, 0.12, 5, 6, 1.8),
    TrendSeries("Khussa", "Unisex", "Footwear", 32, 0.22, 7, 10, 2.0),
    TrendSeries("Streetwear hoodies", "Unisex", "Western", 30, 0.10, 9, 12, 2.3),
    TrendSeries("Jhumka earrings", "Women", "Accessories", 28, 0.20, 7, 10, 2.0),
)

TREND_START = date(2023, 1, 1)
TREND_MONTHS = 45


def _trend_month(offset: int) -> date:
    total = TREND_START.month - 1 + offset
    return date(TREND_START.year + total // 12, total % 12 + 1, 1)


def _demo_trend_frame() -> pd.DataFrame:
    rng = random.Random(20261023)
    rows: list[dict] = []
    for series in TREND_SERIES:
        for offset in range(TREND_MONTHS):
            current = _trend_month(offset)
            month = current.month
            phase = 2 * math.pi * (month - series.peak_month) / 12
            seasonal = series.amplitude * math.cos(phase)
            secondary = 0.3 * series.amplitude * math.sin(2 * phase)
            festive = 3 if month in (11, 12) else 2 if month in (1, 2, 3) else 0
            wedding = 2 if month in (10, 11) else 0
            noise = rng.gauss(0, series.noise)
            value = series.base + series.slope * offset + seasonal + secondary + festive + wedding + noise
            rows.append({
                "date": current,
                "keyword": series.keyword,
                "gender": series.gender,
                "category": series.category,
                "interest": int(round(max(6, min(100, value)))),
            })
    return _prepare_trend_frame(pd.DataFrame(rows))


def filter_audience(frame: pd.DataFrame, audience: str) -> pd.DataFrame:
    if audience == "Women":
        return frame[frame["gender"].isin(["Women", "Unisex"])]
    if audience == "Men":
        return frame[frame["gender"].isin(["Men", "Unisex"])]
    return frame


def _monthly_series(frame: pd.DataFrame) -> pd.Series:
    series = frame.set_index("date")["interest"].sort_index()
    series = series.resample("MS").mean().interpolate(limit_direction="both")
    series = series.ffill().bfill()
    return series.dropna()


def _design_matrix(indices: np.ndarray, length: int) -> np.ndarray:
    centred = (indices - (length - 1) / 2) / max(length - 1, 1)
    columns = [np.ones(len(indices)), centred]
    harmonics = 2 if length >= 24 else 1 if length >= 12 else 0
    for harmonic in range(1, harmonics + 1):
        columns.append(np.sin(2 * np.pi * harmonic * indices / 12))
        columns.append(np.cos(2 * np.pi * harmonic * indices / 12))
    return np.column_stack(columns)


def _forecast(series: pd.Series, horizon: int = HORIZON_MONTHS) -> pd.Series:
    values = series.to_numpy(dtype=float)
    length = len(values)
    if length == 0:
        return pd.Series(dtype=float)

    indices = np.arange(length)
    design = _design_matrix(indices, length)
    penalty = 0.05 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ values)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]

    future_indices = np.arange(length, length + horizon)
    future_design = _design_matrix(future_indices, length)
    predictions = np.clip(future_design @ coefficients, 0, 100)
    future_dates = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    return pd.Series(predictions, index=future_dates, name="forecast")


def build_trend_table(
    frame: pd.DataFrame, audience: str = "Everyone", horizon: int = HORIZON_MONTHS
) -> pd.DataFrame:
    view = filter_audience(frame, audience)
    rows: list[dict] = []

    for (keyword, gender, category), group in view.groupby(
        ["keyword", "gender", "category"], sort=False
    ):
        monthly = _monthly_series(group)
        if len(monthly) < 4:
            continue
        forecast = _forecast(monthly, horizon=horizon)
        forecast_window = float(forecast.iloc[:3].mean())

        comparison_dates = forecast.index[:3] - pd.DateOffset(years=1)
        comparison = (
            monthly.reindex(monthly.index.union(comparison_dates))
            .interpolate(limit_direction="both")
            .reindex(comparison_dates)
        )
        if comparison.notna().all() and float(comparison.mean()) > 0:
            baseline = float(comparison.mean())
        else:
            baseline = float(monthly.iloc[-3:].mean())
        growth = (forecast_window - baseline) / max(baseline, 1.0) * 100

        if growth > 5:
            direction = "Rising"
        elif growth < -5:
            direction = "Cooling"
        else:
            direction = "Steady"

        rows.append({
            "keyword": keyword,
            "gender": gender,
            "category": category,
            "latest": round(float(monthly.iloc[-1])),
            "forecast": round(forecast_window),
            "growth_pct": round(growth),
            "direction": direction,
            "peak_month": forecast.idxmax().strftime("%b %Y"),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("growth_pct", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def forecast_chart_data(
    frame: pd.DataFrame, keyword: str, gender: str, horizon: int = HORIZON_MONTHS
) -> pd.DataFrame:
    group = frame[(frame["keyword"] == keyword) & (frame["gender"] == gender)]
    monthly = _monthly_series(group)
    if monthly.empty:
        return pd.DataFrame(columns=["Month", "Actual", "Forecast"])
    forecast = _forecast(monthly, horizon=horizon)
    chart = pd.DataFrame({"Month": monthly.index, "Actual": monthly.values, "Forecast": np.nan})
    chart.loc[chart.index[-1], "Forecast"] = monthly.iloc[-1]
    future = pd.DataFrame({
        "Month": forecast.index, "Actual": np.nan, "Forecast": forecast.values,
    })
    return pd.concat([chart, future], ignore_index=True).sort_values("Month")


def trend_insight(row: pd.Series) -> str:
    keyword = row["keyword"]
    direction = row["direction"]
    growth = abs(int(row["growth_pct"]))
    peak = row["peak_month"]
    category = str(row["category"]).lower()

    if direction == "Rising":
        opening = f"{keyword} is up about {growth}% against the same months last year, peaking around {peak}."
    elif direction == "Cooling":
        opening = f"{keyword} is down about {growth}% year on year; keep it as a secondary option rather than a lead item."
    else:
        opening = f"{keyword} is holding steady year on year, with its next strong window around {peak}."

    if category in {"eastern", "traditional"}:
        advice = "Line up fabric, embroidery and festive silhouettes for this demand window."
    elif category == "footwear":
        advice = "Stock neutral and gold tones first; they rotate across casual and festive looks."
    elif category == "accessories":
        advice = "Small add-ons carry the trend with less stock risk than a full outfit."
    else:
        advice = "Localise the Western trend with Eastern layering, dupattas or festive footwear."
    return f"{opening} {advice}"


def top_mover(table: pd.DataFrame, direction: str = "Rising") -> pd.Series | None:
    if table.empty:
        return None
    matching = table[table["direction"] == direction]
    if matching.empty:
        return None
    if direction == "Cooling":
        return matching.sort_values("growth_pct").iloc[0]
    return matching.sort_values("growth_pct", ascending=False).iloc[0]

# --------------------------------------------------------------------------
# Streamlit interface
# --------------------------------------------------------------------------

UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; letter-spacing: -0.02em; }
.stApp, [data-testid="stAppViewContainer"] { background: #FBF8F2; }
[data-testid="stSidebar"] { background: #F3EEE4; border-right: 1px solid #E4DAC9; }
[data-testid="stHeader"] { background: rgba(251, 248, 242, 0.85); }
.stTabs [data-baseweb="tab-highlight"] { background-color: #1F6F52; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #1F6F52; }
a { color: #1F6F52; }

.hero {
    background: linear-gradient(115deg, #12392B 0%, #1F6F52 58%, #2F8A67 100%);
    color: #FBF8F2;
    padding: 2.2rem 2.4rem;
    border-radius: 22px;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -70px;
    width: 260px;
    height: 260px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 50%;
}
.hero .eyebrow {
    letter-spacing: 0.2em;
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #EAD6A4;
}
.hero h1 {
    font-size: 2.9rem;
    margin: 0.25rem 0 0.4rem;
    color: #FBF8F2;
    line-height: 1.05;
}
.hero p {
    max-width: 800px;
    font-size: 1rem;
    color: #EAE4D8;
    margin: 0;
}
.hero .tags {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 1.1rem;
}
.hero .tags span {
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
    color: #F3EDE0;
}
.swatch {
    display: inline-block;
    width: 0.72rem;
    height: 0.72rem;
    border-radius: 50%;
    margin-right: 0.4rem;
    border: 1px solid rgba(0, 0, 0, 0.16);
    vertical-align: -1px;
}
.garment-meta {
    font-size: 0.8rem;
    color: #6D6457;
    margin: 0.2rem 0 0.45rem;
}
[data-testid="stMetric"] {
    background: #F3EEE4;
    border: 1px solid #E4DAC9;
    padding: 0.8rem 1rem;
    border-radius: 16px;
}
[data-testid="stMetricLabel"] { color: #6D6457; }
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 1rem;
}
.kpi-card {
    background: #F3EEE4;
    border: 1px solid #E4DAC9;
    border-radius: 16px;
    padding: 0.9rem 1rem;
}
.kpi-label { color: #6D6457; font-size: 0.78rem; font-weight: 500; }
.kpi-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.45rem;
    line-height: 1.2;
    color: #23352C;
    margin: 0.28rem 0 0.2rem;
    overflow-wrap: anywhere;
}
.kpi-foot { font-size: 0.78rem; color: #6D6457; }
.kpi-up { color: #1F6F52; font-weight: 600; }
.kpi-down { color: #A6432F; font-weight: 600; }
@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.small-note { color: #6D6457; font-size: 0.86rem; }
</style>
"""

HERO = """
<div class="hero">
    <div class="eyebrow">Pakistan &middot; Men &amp; Women</div>
    <h1>Libas Lab</h1>
    <p>Build outfits from the clothes you already own, finish them with accessories that fit
    a Pakistani occasion, and see which styles are gaining momentum before the season turns.</p>
    <div class="tags">
        <span>FashionCLIP garment vision</span>
        <span>Accessory styling</span>
        <span>Pinterest look ideas</span>
        <span>Six-month trend forecast</span>
    </div>
</div>
"""


@st.cache_resource(show_spinner=False)
def get_encoder() -> FashionEncoder:
    return FashionEncoder()


@st.cache_data(show_spinner=False)
def load_sample_trends() -> pd.DataFrame:
    return _demo_trend_frame()


@st.cache_data(show_spinner=False)
def load_uploaded_trends(payload: bytes) -> pd.DataFrame:
    return load_trend_data(io.BytesIO(payload))


def init_state() -> None:
    if "wardrobe" not in st.session_state:
        st.session_state.wardrobe = {
            gender: {"top": [], "bottom": []} for gender in GENDER_OPTIONS
        }
    if "dismissed" not in st.session_state:
        st.session_state.dismissed = {
            gender: {"top": set(), "bottom": set()} for gender in GENDER_OPTIONS
        }
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = {gender: 0 for gender in GENDER_OPTIONS}
    if "gender" not in st.session_state:
        st.session_state.gender = "Women"
    if "season" not in st.session_state:
        st.session_state.season = "Winter" if date.today().month in {11, 12, 1, 2} else "Summer"
    st.session_state.setdefault("model_status", "")


def _digest(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _load_encoder() -> FashionEncoder | None:
    try:
        with st.spinner("Loading FashionCLIP weights. This happens once."):
            encoder = get_encoder()
        st.session_state.model_status = f"FashionCLIP ready on {encoder.device_label}"
        return encoder
    except ModelLoadError as exc:
        st.session_state.model_status = "error"
        st.error(str(exc))
        return None


def _ingest_uploads(
    files, slot: str, gender: str, store: dict, encoder: FashionEncoder,
    known_ids: set[str], dismissed_ids: set[str],
) -> int:
    images = []
    metadata = []
    for uploaded in files:
        payload = uploaded.getvalue()
        digest = _digest(payload)
        if digest in known_ids or digest in dismissed_ids:
            continue
        try:
            with Image.open(io.BytesIO(payload)) as raw:
                raw.load()
                image = raw.convert("RGB")
        except Exception:
            st.warning(f"Skipped {uploaded.name}: the file could not be read as an image.")
            continue
        images.append(image)
        metadata.append((digest, uploaded.name, image))
        known_ids.add(digest)

    if not images:
        return 0

    analyses = analyze_garments(images, slot, gender, encoder)
    for (digest, name, image), (attributes, embedding) in zip(metadata, analyses):
        store[slot].append({
            "id": digest,
            "name": name,
            "slot": slot,
            "gender": gender,
            "image": image_to_jpeg(image),
            "embedding": embedding,
            "attrs": attributes,
        })
    return len(analyses)


def _pending_files(files, known_ids: set[str], dismissed_ids: set[str]) -> list:
    pending = []
    for uploaded in files or []:
        digest = _digest(uploaded.getvalue())
        if digest not in known_ids and digest not in dismissed_ids:
            pending.append(uploaded)
    return pending


def _render_item_grid(items: list[dict], slot: str, gender: str) -> None:
    label = "tops" if slot == "top" else "bottoms"
    if not items:
        st.info(f"No {label} yet. Upload a few above to start matching.")
        return
    columns = st.columns(4)
    for index, item in enumerate(items):
        with columns[index % 4]:
            st.image(item["image"], width="stretch")
            attributes = item["attrs"]
            st.markdown(f"**{attributes['type']}**")
            st.markdown(
                '<div class="garment-meta">'
                f'<span class="swatch" style="background:{attributes["color_hex"]}"></span>'
                f"{attributes['color_name']} &middot; {attributes['pattern']} &middot; "
                f"{attributes['formality_label']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{attributes['style']} · {attributes['type_confidence']}% type confidence")
            if st.button("Remove", key=f"remove-{gender}-{slot}-{item['id']}", width="stretch"):
                items.remove(item)
                st.session_state.dismissed[gender][slot].add(item["id"])
                st.rerun()


def _wardrobe_table(store: dict) -> pd.DataFrame:
    rows = []
    for slot, piece in (("top", "Top"), ("bottom", "Bottom")):
        for item in store[slot]:
            attributes = item["attrs"]
            rows.append({
                "Item": item["name"],
                "Piece": piece,
                "Detected type": attributes["type"],
                "Colour": attributes["color_name"],
                "Pattern": attributes["pattern"],
                "Style": attributes["style"],
                "Formality": attributes["formality_label"],
                "Confidence": f"{attributes['type_confidence']}%",
            })
    return pd.DataFrame(rows)


def render_wardrobe(gender: str) -> None:
    store = st.session_state.wardrobe[gender]
    nonce = st.session_state.uploader_nonce[gender]

    st.markdown("## My wardrobe")
    st.caption(
        "Upload one garment per photo, ideally laid flat on a plain background. "
        "FashionCLIP reads the piece and tags its type, pattern, colour and formality. "
        "The cupboard is kept separately for women and men."
    )

    top_column, bottom_column = st.columns(2)
    with top_column:
        top_files = st.file_uploader(
            "Tops — shirts, kurtas, kurtis, blouses",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"upload-top-{gender}-{nonce}",
            help="JPG, PNG or WEBP. Upload several pieces at once.",
        )
    with bottom_column:
        bottom_files = st.file_uploader(
            "Bottoms — trousers, jeans, shalwar, skirts",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"upload-bottom-{gender}-{nonce}",
            help="JPG, PNG or WEBP. Upload several pieces at once.",
        )

    known_ids = {item["id"] for slot in ("top", "bottom") for item in store[slot]}
    dismissed_top = st.session_state.dismissed[gender]["top"]
    dismissed_bottom = st.session_state.dismissed[gender]["bottom"]
    pending_top = _pending_files(top_files, known_ids, dismissed_top)
    pending_bottom = _pending_files(
        bottom_files, known_ids | {_digest(f.getvalue()) for f in pending_top}, dismissed_bottom
    )

    if pending_top or pending_bottom:
        encoder = _load_encoder()
        if encoder is not None:
            added = 0
            try:
                with st.spinner("FashionCLIP is reading your wardrobe..."):
                    if pending_top:
                        added += _ingest_uploads(
                            pending_top, "top", gender, store, encoder, known_ids, dismissed_top
                        )
                    if pending_bottom:
                        added += _ingest_uploads(
                            pending_bottom, "bottom", gender, store, encoder, known_ids, dismissed_bottom
                        )
            except Exception as exc:
                st.error(f"Garment analysis failed: {exc}")
            else:
                if added:
                    st.success(f"Added {added} garment{'s' if added != 1 else ''} to your cupboard.")

    top_items = store["top"]
    bottom_items = store["bottom"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Tops", len(top_items))
    metric_columns[1].metric("Bottoms", len(bottom_items))
    metric_columns[2].metric("Possible pairings", len(top_items) * len(bottom_items))

    if top_items or bottom_items:
        clear_column, note_column = st.columns([1, 3])
        with clear_column:
            if st.button("Clear this wardrobe", width="stretch"):
                store["top"].clear()
                store["bottom"].clear()
                st.session_state.dismissed[gender] = {"top": set(), "bottom": set()}
                st.session_state.uploader_nonce[gender] += 1
                st.rerun()
        with note_column:
            st.caption(
                "Removing a garment here hides it from the current session; the file stays in the uploader."
            )

    st.markdown("### Tops")
    _render_item_grid(top_items, "top", gender)
    st.markdown("### Bottoms")
    _render_item_grid(bottom_items, "bottom", gender)

    if top_items or bottom_items:
        with st.expander("Cupboard table"):
            st.dataframe(_wardrobe_table(store), hide_index=True, width="stretch")


def _flatten_items(items: list[dict]) -> list[dict]:
    return [
        {**item["attrs"], "id": item["id"], "name": item["name"], "image": item["image"]}
        for item in items
    ]


def render_studio(gender: str, season: str) -> None:
    store = st.session_state.wardrobe[gender]
    st.markdown("## Outfit studio")
    st.caption(
        "Every top and bottom in your cupboard is paired and scored on colour harmony, "
        "occasion formality, pattern balance and Eastern/Western coherence. "
        "Accessories are then matched from a Pakistani catalog."
    )

    if not store["top"] or not store["bottom"]:
        st.info(
            "Upload at least one top and one bottom in **My wardrobe** to generate looks. "
            "FashionCLIP tags the garments, then this tab does the styling."
        )
        return

    controls = st.columns([1.3, 1, 1])
    with controls[0]:
        occasion_label = st.selectbox(
            "Occasion",
            [occasion.label for occasion in OCCASIONS],
            help="The target formality changes the scoring and accessory matches.",
        )
    with controls[1]:
        limit = st.slider("Looks to show", min_value=3, max_value=8, value=5)
    with controls[2]:
        st.metric("Pairings scored", len(store["top"]) * len(store["bottom"]))

    occasion = next(item for item in OCCASIONS if item.label == occasion_label)
    tops = _flatten_items(store["top"])
    bottoms = _flatten_items(store["bottom"])
    all_looks, total = build_outfits(
        tops, bottoms, gender, occasion.key, season, limit=None
    )
    looks = all_looks[:limit]

    st.caption(
        f"Showing the top {len(looks)} of {total} pairings for a {season.lower()} "
        f"{occasion.label.lower()} look."
    )

    for rank, look in enumerate(looks, start=1):
        top = look["top"]
        bottom = look["bottom"]
        with st.container(border=True):
            image_columns = st.columns([1, 1, 2.4])
            with image_columns[0]:
                st.image(top["image"], width="stretch")
                st.caption(f"Top · {top['color_name']} {top['type']}")
            with image_columns[1]:
                st.image(bottom["image"], width="stretch")
                st.caption(f"Bottom · {bottom['color_name']} {bottom['type']}")
            with image_columns[2]:
                st.markdown(f"#### Look {rank} · {look['score']}% match")
                st.progress(look["score"], text="Colour, formality and texture balance")
                for reason in look["reasons"]:
                    st.markdown(f"- {reason}")
                st.link_button(
                    "Find this look on Pinterest",
                    look["pinterest_url"],
                    key=f"pinterest-look-{rank}-{top['id'][:8]}-{bottom['id'][:8]}",
                    width="stretch",
                )

            if look["accessories"]:
                st.markdown("**Finish it with**")
                accessory_columns = st.columns(len(look["accessories"]))
                for index, (column, accessory) in enumerate(
                    zip(accessory_columns, look["accessories"])
                ):
                    with column:
                        st.markdown(f"**{accessory['name']}**")
                        st.caption(accessory["group"])
                        st.write(accessory["why"])
                        st.link_button(
                            "Pinterest ideas",
                            accessory["pinterest_url"],
                            key=f"pinterest-accessory-{rank}-{index}-{top['id'][:6]}",
                            width="stretch",
                        )

    with st.expander("All scored combinations"):
        rows = [
            {
                "Rank": index,
                "Top": f"{look['top']['color_name']} {look['top']['type']}",
                "Bottom": f"{look['bottom']['color_name']} {look['bottom']['type']}",
                "Match %": look["score"],
            }
            for index, look in enumerate(all_looks, start=1)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_trends() -> None:
    st.markdown("## Trend radar")
    st.caption(
        "Search-interest data is used as a demand signal: each style is seasonally "
        "forecast six months ahead and ranked by momentum. Men, women and unisex styles "
        "are included."
    )

    with st.expander("Use your own trend data (CSV)"):
        st.markdown(
            "Upload a long-format CSV with **date, keyword, gender, category, interest** "
            "(interest 0–100). A Google Trends export can be reshaped into this format."
        )
        st.code(
            "date,keyword,gender,category,interest\n"
            "2026-01-01,Co-ord sets,Women,Western,42\n"
            "2026-01-01,Linen kurtas,Men,Eastern,31",
            language="csv",
        )
        uploaded = st.file_uploader("Upload trend CSV", type=["csv"], key="trend-upload")
        demo_frame = load_sample_trends()
        st.download_button(
            "Download the demo CSV",
            data=demo_frame.to_csv(index=False).encode("utf-8"),
            file_name="pakistan_trends_sample.csv",
            mime="text/csv",
        )

    uploaded_payload = uploaded.getvalue() if uploaded is not None else None

    if uploaded_payload is not None:
        version = hashlib.md5(uploaded_payload).hexdigest()[:8]
        try:
            frame = load_uploaded_trends(uploaded_payload)
            source_label = uploaded.name
            is_demo = False
        except TrendDataError as exc:
            st.error(f"{exc} Falling back to the demo data.")
            frame = load_sample_trends()
            source_label = "Generated demo dataset"
            is_demo = True
            version = "demo"
    else:
        frame = load_sample_trends()
        source_label = "Generated demo dataset"
        is_demo = True
        version = "demo"

    if is_demo:
        st.info(
            "You are viewing the synthetic demo dataset generated by the app. "
            "Upload a real trend CSV above to forecast actual market signals."
        )
    st.caption(f"Data source: {source_label} · {len(frame)} observations")

    filters = st.columns([1, 2])
    with filters[0]:
        audience = st.radio("Audience", AUDIENCES, horizontal=True)
    categories = sorted(frame["category"].dropna().unique().tolist())
    with filters[1]:
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=categories,
            key=f"trend-categories-{version}",
        )
    st.caption(
        "Choosing Women or Men also includes unisex styles. "
        f"Forecast horizon: {HORIZON_MONTHS} months from the last observation."
    )

    if not selected_categories:
        st.info("Select at least one category to see the forecast.")
        return

    filtered = frame[frame["category"].isin(selected_categories)]
    table = build_trend_table(filtered, audience=audience)
    if table.empty:
        st.warning(
            "Not enough observations per keyword to fit a forecast. "
            "Provide at least four monthly points for each keyword and gender."
        )
        return

    top_signal = table.iloc[0]
    rising_count = int((table["direction"] == "Rising").sum())
    cooling = top_mover(table, "Cooling")
    last_date = pd.to_datetime(frame["date"]).max()
    window_start = last_date + pd.offsets.MonthBegin(1)
    window_end = window_start + pd.DateOffset(months=HORIZON_MONTHS - 1)

    top_delta = int(top_signal["growth_pct"])
    if cooling is not None:
        cooling_foot = (
            f'<div class="kpi-foot kpi-down">↓ {int(cooling["growth_pct"]):+d}% '
            "year on year</div>"
        )
    else:
        cooling_foot = '<div class="kpi-foot">No style is cooling sharply</div>'
    cooling_value = escape(str(cooling["keyword"])) if cooling is not None else "None flagged"

    kpi_html = textwrap.dedent(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Top rising signal</div>
                <div class="kpi-value">{escape(str(top_signal["keyword"]))}</div>
                <div class="kpi-foot kpi-up">↑ {top_delta:+d}% year on year</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Rising styles</div>
                <div class="kpi-value">{rising_count} of {len(table)}</div>
                <div class="kpi-foot">tracked styles gaining</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Cooling watch-out</div>
                <div class="kpi-value">{cooling_value}</div>
                {cooling_foot}
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Forecast window</div>
                <div class="kpi-value">{HORIZON_MONTHS} months</div>
                <div class="kpi-foot">{window_start:%b %Y} – {window_end:%b %Y}</div>
            </div>
        </div>
        """
    ).strip()
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown("### Forecast detail")
    options = list(table.index)
    selected_index = st.selectbox(
        "Inspect a trend",
        options,
        format_func=lambda index: (
            f"{table.loc[index, 'keyword']} · "
            f"{table.loc[index, 'gender']} · {table.loc[index, 'category']}"
        ),
        label_visibility="collapsed",
    )
    selected_row = table.loc[selected_index]
    chart = forecast_chart_data(frame, selected_row["keyword"], selected_row["gender"])
    if not chart.empty:
        st.line_chart(
            chart.set_index("Month")[["Actual", "Forecast"]],
            height=320,
            color=["#1F6F52", "#C9A227"],
        )
        st.caption(
            "Green is observed interest; gold is the seasonal forecast. "
            f"Peak forecast month: {selected_row['peak_month']}. "
            "The model is directional and assumes history repeats its seasonal shape."
        )

    st.markdown("### Ranked trends")
    st.caption(
        "Momentum compares the next three forecast months with the same three months a "
        "year earlier, so a routine seasonal dip is not mistaken for a fading trend."
    )
    display = table[
        ["rank", "keyword", "gender", "category", "latest", "forecast", "growth_pct", "direction", "peak_month"]
    ].rename(columns={
        "rank": "Rank", "keyword": "Trend", "gender": "Audience", "category": "Category",
        "latest": "Current", "forecast": "Next 3 mo", "growth_pct": "Momentum",
        "direction": "Signal", "peak_month": "Peak month",
    })
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Momentum": st.column_config.NumberColumn(format="%d%%"),
            "Current": st.column_config.NumberColumn(format="%d"),
            "Next 3 mo": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.markdown("### What it means for a Pakistani wardrobe")
    insight_rows = [row for _, row in table.head(3).iterrows()]
    if cooling is not None and cooling.name not in table.head(3).index:
        insight_rows.append(cooling)
    for row in insight_rows:
        with st.container(border=True):
            columns = st.columns([1, 2.4])
            with columns[0]:
                st.markdown(f"**{row['keyword']}**")
                st.caption(f"{row['category']} · {row['gender']} · {row['direction']}")
            with columns[1]:
                st.write(trend_insight(row))

    with st.expander("How the forecast works"):
        st.markdown(
            "Each keyword's monthly search interest is fitted with a **ridge regression "
            "with a linear trend and yearly Fourier seasonality**, then projected six "
            "months forward. Momentum compares the next three forecast months with the "
            "same three months a year earlier, which strips out normal seasonal swings. "
            "It is intentionally lightweight and transparent: no black-box LLM guesses, "
            "and it gets more reliable as you feed it real, longer Google Trends history. "
            "Eid, wedding season and lawn season are captured by the seasonal signal "
            "rather than hard-coded dates."
        )


def main() -> None:
    st.set_page_config(
        page_title="Libas Lab | Pakistani outfit and trend studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Style profile")
        gender = st.radio("Shopping for", GENDER_OPTIONS, key="gender")
        season = st.selectbox("Season", SEASON_OPTIONS, key="season")
        st.caption(
            "The wardrobe and outfit studio follow this profile. The trend radar has "
            "its own audience filter."
        )
        st.divider()
        st.markdown("### Fashion engine")
        status = st.session_state.model_status
        if status == "error":
            st.warning("FashionCLIP did not load. Uploading will retry.")
        elif status:
            st.caption(status)
        else:
            st.caption("FashionCLIP loads on your first upload, then stays cached.")
        st.caption(
            "Model: [patrickjohncyh/fashion-clip]"
            "(https://huggingface.co/patrickjohncyh/fashion-clip) · "
            "first run downloads the weights."
        )
        st.divider()
        st.markdown("### How to get the best results")
        st.caption(
            "Shoot garments flat on a plain background in daylight. One piece per photo. "
            "Makeup-heavy campaign images reduce attribute accuracy."
        )

    wardrobe_tab, studio_tab, trends_tab = st.tabs(
        ["My wardrobe", "Outfit studio", "Trend radar"]
    )
    with wardrobe_tab:
        render_wardrobe(gender)
    with studio_tab:
        render_studio(gender, season)
    with trends_tab:
        render_trends()

    st.divider()
    st.caption(
        "Libas Lab · FashionCLIP for garment recognition · rule-based styling for "
        "explainable matches · seasonal ridge regression for demand forecasts · "
        "Pinterest buttons open a search in a new tab. Trend demo data is synthetic; "
        "forecasts are directional, not guarantees."
    )


if __name__ == "__main__":
    main()
