# PDF-Rename
# Invoice Processor 🔍🧾  
Ein Python-Skript zum automatisierten Aufbereiten, Analysieren und Ablegen von Rechnungen (PDF & Bilder).  
Mehrseitige PDFs werden aufgetrennt, per OCR ausgelesen, nach Lieferant*innen erkannt und anschließend sinnvoll umbenannt und in fertige bzw. unsichere Ordner verschoben.

---

## Features

| Feature | Erklärung |
|---------|-----------|
| **Mehrseitige PDF-Erkennung** | Erkennt automatisch, ob eine Datei mehrere Rechnungen enthält und trennt diese bei Bedarf. |
| **OCR-Unterstützung (Tesseract)** | Nutzt zuerst den eingebetteten Textlayer; fällt dann auf Tesseract-OCR (Deutsch) zurück. |
| **Ähnlichkeits-Analyse** | Header, Footer, Ränder & Text-Content werden gewichtet, um Seiten derselben Rechnung zusammenzuführen (`header_similarity`, `footer_similarity`, …). |
| **Lieferanten-Erkennung** | Verwendet GPT-4 (oder das in `config.yaml` angegebene Modell), um Firmenname, Rechnungsnummer, Datum & Betrag zu extrahieren. |
| **Intelligentes Dateinamen-Schema** |  `YYYY-MM-DD_<Firma>_<RechnungsNr>_<Betrag>.pdf`  – fehlende Infos werden mit Platzhaltern ersetzt. |
| **Konfigurierbare Ordnerstruktur** | Standard: `Rechnungen/`, `Rechnungen-Unsicher/`, `Backup/`, `temp_invoices/` (alles anpassbar). |
| **CLI-Schalter** | Aggressives/konservatives Splitten, Schwellenwerte, Bundle-Modus u.v.m. |
| **GUI-Fallback** | Bei kritischen Fehlern erscheint ein kleines Tkinter-Popup. |

---

## Voraussetzungen

| Software | Version / Hinweis |
|----------|------------------|
| Python | ≥ 3.8 |
| [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) | inkl. Sprachpaket **`deu`** |
| [Poppler](https://poppler.freedesktop.org/) | Nur benötigt, wenn `pdf2image` nicht schon ein Poppler-Binary findet |

### Python-Pakete

Installiere die Abhängigkeiten am einfachsten via:

```bash
pip install -r requirements.txt
