# PharmIQ API Reference

Base URL: `https://<your-azure-fqdn>` | Local: `http://localhost:8000`

---

## Endpoints

### `GET /health`
Returns model load status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /api/v1/predict`
Predict price tier for a single medicine.

**Request body:**
```json
{
  "salt_composition": "Paracetamol (500mg) + Ibuprofen (400mg)",
  "dosage_form": "tablet",
  "pack_size_units": 10,
  "salt_count": 2,
  "manufacturer_tier": 1,
  "max_dose_mg": 500,
  "is_branded": 0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `salt_composition` | string | ✅ | Full active ingredient string |
| `dosage_form` | string | ✅ | One of: tablet, capsule, liquid, injection, topical, drops, spray, inhaler, powder, patch, suppository, other |
| `pack_size_units` | float > 0 | ✅ | Pack size in units/ml/gm |
| `salt_count` | int 1–10 | ✅ | Number of active ingredients |
| `manufacturer_tier` | int 0–2 | ✅ | 0=generic, 1=mid, 2=top-tier |
| `max_dose_mg` | float ≥ 0 | optional | Largest dose in mg (default 0) |
| `is_branded` | int 0–1 | optional | 1=branded name (default 0) |

**Response:**
```json
{
  "price_tier": "Mid",
  "tier_code": 1,
  "probabilities": {
    "Budget": 0.21,
    "Mid": 0.48,
    "Premium": 0.19,
    "Luxury": 0.12
  },
  "interpretation": "Priced between the 25th and 50th percentile for this dosage form.",
  "model_version": "v1"
}
```

**Error responses:**

| Code | Reason |
|------|--------|
| 422 | Validation error (invalid field values) |
| 503 | Model not loaded |
| 500 | Internal prediction error |

---

### `GET /api/v1/tiers`
Returns tier metadata and supported dosage forms.

**Response:**
```json
{
  "tiers": [
    {"code": 0, "label": "Budget", "description": "Bottom 25% within dosage form group"},
    {"code": 1, "label": "Mid", "description": "25th-50th percentile within dosage form group"},
    {"code": 2, "label": "Premium", "description": "50th-75th percentile within dosage form group"},
    {"code": 3, "label": "Luxury", "description": "Top 25% within dosage form group"}
  ],
  "dosage_forms": ["tablet", "capsule", "liquid", "injection", "topical", "drops", "spray", "inhaler", "powder", "patch", "suppository", "other"],
  "manufacturer_tier_guide": {
    "0": "Generic / small manufacturer",
    "1": "Mid-tier pharma (Micro Labs, Macleods, FDC, etc.)",
    "2": "Top-tier pharma (Sun Pharma, Cipla, Lupin, Abbott, Pfizer, etc.)"
  }
}
```

---

## Example — cURL

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "salt_composition": "Azithromycin (500mg)",
    "dosage_form": "tablet",
    "pack_size_units": 3,
    "salt_count": 1,
    "manufacturer_tier": 2,
    "max_dose_mg": 500,
    "is_branded": 1
  }'
```

## Example — Python

```python
import requests

payload = {
    "salt_composition": "Metformin (500mg) + Glimepiride (1mg)",
    "dosage_form": "tablet",
    "pack_size_units": 10,
    "salt_count": 2,
    "manufacturer_tier": 1,
    "max_dose_mg": 500,
    "is_branded": 0,
}

resp = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(resp.json())
```

---

## Interactive Docs

Visit `/docs` (Swagger UI) or `/redoc` for full interactive API documentation.
