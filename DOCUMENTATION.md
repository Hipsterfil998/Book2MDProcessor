# Documentazione della Codebase — Pipeline di Conversione Libri

## Panoramica

La pipeline converte libri in formato PDF e EPUB in Markdown strutturato, ne estrae metadati bibliografici e di genere, valuta la qualità della conversione e produce annotazioni di dependency parsing. Il tutto è orchestrato da un'unica classe centrale (`ConverterPipeline`) e configurato da un singolo file `config.py`.

---

## Struttura del progetto

```
.
├── config.py                        # Configurazione centralizzata
├── book_converter.py                # Orchestratore della pipeline
├── utils.py                         # Utilità condivise
│
├── converters/
│   ├── text_extraction.py           # Conversione rule-based (senza LLM)
│   ├── pdf2md_LLM.py                # PDF → Markdown via Qwen2.5-VL
│   └── epub2md_LLM.py               # EPUB → Markdown via Qwen2.5
│
├── metadata/
│   └── metadata_extractor.py        # Estrazione autore/titolo/anno/genere
│
├── quality_evaluation/
│   └── evaluator.py                 # LLM-as-judge per valutazione fedeltà
│
├── dependency_parsing/
│   └── dependency_parsing.py        # Annotazione linguistica con Stanza
│
├── books/                           # Input: PDF e EPUB originali
├── output/                          # Output: Markdown + eval pages/chunks
├── scores/                          # Score JSON per libro
│
└── tests/
    ├── test_book_converter.py
    ├── test_metadata.py
    └── test_utils.py
```

---

## Flusso della pipeline

```
books/                     output/{stem}/
  *.pdf  ──► PDFConverter ──► {stem}.md
                          ──► eval_pages/{i}.png + {i}.md
                          ──► images/
  *.epub ──► EPUBConverter ──► {stem}.md
                           ──► eval_chunks/{i}.html + {i}.md

output/                    scores/
  **/eval_pages/ ──► QualityEvaluator ──► {book}_scores.json
  **/eval_chunks/

output/                    metadata/metadata.csv
  **/eval_pages/  ──► MetadataExtractor ──► author, title, year, genre
  **/eval_chunks/

output/
  **/*.md ──► DependencyParser ──► parsed_output/*.conllu / *.json
```

---

## `config.py` — Configurazione centralizzata

Tutto ciò che è parametrico (modelli, prompt, path, token limits) è definito qui. Non ci sono costanti magiche sparse nel codice.

**Modelli usati:**
- `PDF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"` — modello vision-language per PDF (legge immagini di pagine)
- `TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"` — modello testo per EPUB e metadati

**Scelta dei modelli:** Qwen2.5 è open-weight, disponibile su HuggingFace, con ottimo supporto per italiano e tedesco (lingue del corpus). La versione VL (Vision-Language) è necessaria per il PDF perché la pipeline rasterizza le pagine in PNG invece di usare l'estrazione testuale diretta — questo permette di preservare layout, formule e immagini.

**`ENABLE_PREFIX_CACHING = True`:** abilita l'Automatic Prefix Caching di vLLM. Siccome tutti i libri in un batch condividono lo stesso system prompt, vLLM può cachearlo nei KV states e non ricalcolarlo per ogni richiesta. I prompt di sistema (`BIBLIO_PROMPT`, `GENRE_PROMPT`, `PDF_PROMPT`) sono costanti appositamente per massimizzare il cache hit rate.

**`_StderrFilter`:** filtra i messaggi di errore non-fatali di protobuf/grpc che vLLM emette su stderr — rumore che confonde i log di produzione.

---

## `utils.py` — Utilità condivise

### `pil_to_data_url(img)`
Converte un'immagine PIL in data URL base64 (`data:image/png;base64,...`). Usata per passare le pagine PNG al modello vision-language tramite vLLM che accetta il formato multimodale OpenAI-compatible.

### `sample_indices(total, n=20)`
Seleziona `n` indici di pagina per l'evaluation, con questa strategia ibrida:

1. **Garantiti:** le prime `min(10, n)` pagine sono sempre incluse — perché contengono frontespizio, colophon e metadati
2. **Stratificati:** i restanti slot vengono riempiti con campionamento casuale dal corpo (~75%) e dal fondo (~25%) del documento

**Perché stratificato e non uniforme?** Un campionamento puramente uniforme sovra-rappresenta il corpo e ignora le pagine finali (indici, colofoni). La proporzione 75/25 body/back empiricamente copre meglio la variabilità tipografica.

### `suppress_worker_stderr()`
Context manager che intercetta fd 2 a livello OS (non solo `sys.stderr`) per filtrare il rumore di protobuf. I worker vLLM vengono forkati e ereditano fd 2 direttamente — un wrapper Python non sarebbe sufficiente.

---

## `book_converter.py` — Orchestratore

`ConverterPipeline` espone tre metodi di conversione:

| Metodo | Converter usato | Quando usarlo |
|--------|----------------|---------------|
| `run_simple()` | `DocumentProcessor` (rule-based) | Bozza rapida, nessun GPU |
| `run_pdf_llm()` | `PDFToMarkdownConverter` | PDF con layout complesso |
| `run_epub_llm()` | `EpubToMarkdownConverter` | EPUB |

**Resume/idempotenza:** `_already_converted(stem)` controlla se `output/{stem}/{stem}.md` esiste. Se sì, il libro viene saltato. Questo permette di interrompere e riprendere la pipeline senza riprocessare libri già convertiti — fondamentale quando si lavora con corpus di centinaia di libri e l'inferenza LLM è costosa.

---

## `converters/text_extraction.py` — Conversione rule-based

`DocumentProcessor` converte senza LLM usando regole deterministiche.

**PDF:** usa PyMuPDF (`fitz`) per estrarre il testo in modalità `rawdict`, che espone font size e flags per ogni span. Le regole di mappatura sono:
- font size ≥ 22 → `# H1`, ≥ 18 → `## H2`, ≥ 14 → `### H3`, ≥ 12 + bold → `#### H4`
- flags & 16 = bold → `**testo**`, flags & 2 = italic → `*testo*`
- font size < 9 e inizia con `\d*†‡§` → nota a piè di pagina `> [^fn]: ...`
- match `^(figura|fig|tabella|tab)` → didascalia `*testo*`
- immagini estratte in `images/` e inserite come `![caption](images/filename)`

**EPUB:** usa `ebooklib` per estrarre i capitoli HTML e un walker ricorsivo HTML→Markdown che gestisce: intestazioni, bold/italic, code block, liste ordinate/non ordinate, tabelle GFM, footnote (`epub:type="footnote"`), figure con didascalia, blockquote, callout box (riconosciuti da CSS class: `callout|note|warning|tip`).

**Limitazione:** questo approccio rule-based fallisce con PDF scansionati, formule LaTeX complesse e layout a colonne multiple — da qui la necessità dei converter LLM.

---

## `converters/pdf2md_LLM.py` — PDF → Markdown via LLM

`PDFToMarkdownConverter` usa Qwen2.5-VL (vision-language):

1. Rasterizza ogni pagina PDF in PNG a `PDF_DPI=300` dpi
2. Codifica ogni PNG come data URL base64
3. Costruisce un batch di messaggi multimodali: `[{"type": "image_url", ...}, {"type": "text", "text": PDF_PROMPT}]`
4. Inferenza batch con `LLM.chat()` — tutte le pagine del libro in un'unica chiamata
5. Salva il Markdown risultante
6. Salva le pagine campionate in `eval_pages/{i}.png` + `eval_pages/{i}.md` per evaluation e metadati

**Perché rasterizzare invece di estrarre testo?** L'estrazione testuale diretta perde la struttura visiva (colonne, tabelle, equazioni). Con la rasterizzazione, il modello vede esattamente la pagina come un lettore umano.

**Batching:** tutta la pipeline beneficia del batching nativo di vLLM che esegue inferenza parallela su tutte le pagine con un unico forward pass (continuous batching). Questo è molto più efficiente che elaborare le pagine una per volta.

---

## `converters/epub2md_LLM.py` — EPUB → Markdown via LLM

`EpubToMarkdownConverter` usa Qwen2.5 (testo puro):

1. Converte l'EPUB in HTML con `pypandoc`
2. Estrae le immagini in `images/` e riscrive i `src` delle `<img>` con i path locali
3. Divide l'HTML in chunk da max `EPUB_MAX_CHUNK_CHARS=8000` caratteri per top-level block tag (`section`, `article`, `div`)
4. Inferenza batch di tutti i chunk con il prompt `EPUB_PROMPT`
5. Join dei chunk con `\n\n---\n\n`
6. Salva chunk campionati in `eval_chunks/{i}.html` + `eval_chunks/{i}.md`

**Perché chunking?** I modelli hanno una context window limitata. Spezzare per top-level tag garantisce che ogni chunk sia semanticamente coeso (un blocco HTML completo, non spezzato a metà frase).

**Differenza da `run_simple`:** il LLM gestisce meglio elementi complessi come tabelle irregolari, liste annidate e footnote con riferimenti incrociati.

---

## `metadata/metadata_extractor.py` — Estrazione metadati

`MetadataExtractor` estrae 5 campi: autore, titolo, anno, genere — tutto con zero-shot prompting sullo stesso modello testuale.

### `collect_samples(output_dir)`
Per ogni libro cerca `eval_pages/` (preferito) o `eval_chunks/` e seleziona:
- `front_files = guaranteed[:5]` — pagine 0-4: frontespizio, colophon, prefazione → contengono autore/titolo/anno
- `body_files = guaranteed[5:][-3:]` — pagine 7-9 delle prime 10: superato il fronte materia ma ancora presto nel libro → campione rappresentativo per il genere

**Perché queste pagine?** Le prime 5 pagine di un libro contengono quasi sempre le informazioni bibliografiche. Le pagine 7-9 sono post-fronte-materia e rappresentano il tono/stile del testo senza essere ancora troppo specialistiche. Essendo tutte nell'insieme garantito dei primi 10, sono sempre disponibili indipendentemente dalla lunghezza del libro.

### `run(output_dir, output_csv)`
Esegue due inferenze batch separate:
1. `BIBLIO_PROMPT` + `"Book filename: {name}\n\n{front_text}"` → JSON `{author, title, year}`
2. `GENRE_PROMPT` + `body_text` → JSON `{genre}`

**Resume CSV:** se `output_csv` esiste già, carica i record esistenti, filtra i libri già processati e appende solo i nuovi. Se non c'è nulla di nuovo, ritorna senza riscrivere il file (preserva `mtime`).

**Perché due batch separati?** I due prompt sono molto diversi (biblio vs genere) e richiedono testi di input diversi. Tenerli separati massimizza il prefix cache hit rate: all'interno di ogni batch, tutti i messaggi condividono lo stesso system prompt costante.

**`_parse_json(raw)`:** gestisce output imperfetti del modello. Prima tenta `json.loads()` diretto. Se fallisce, tenta il recovery: prende il testo prima di una riga vuota (il modello a volte aggiunge testo dopo il JSON) e aggiunge `}` mancante se necessario.

---

## `quality_evaluation/evaluator.py` — Valutazione qualità

`QualityEvaluator` implementa il pattern **LLM-as-judge**: usa un modello linguistico per valutare l'output di un altro.

### `evaluate_pdf(eval_pages_dir, scores_dir)`
Per ogni coppia `{i}.png` + `{i}.md`:
- Passa l'immagine originale + il Markdown generato al judge
- Il judge valuta la fedeltà su 3 dimensioni: `text` (1-5), `structure` (1-5), `math` (1-5)
- Output: JSON con media e score per pagina

### `evaluate_epub(eval_chunks_dir, scores_dir)`
Per ogni coppia `{i}.html` + `{i}.md`:
- Valuta fedeltà su 2 dimensioni: `text` e `structure` (senza `math` perché l'HTML non ha formule)

**Vantaggio del design eval-per-book:** la valutazione non richiede accesso ai file originali — usa solo le coppie sorgente/output già salvate durante la conversione. Questo disaccoppia la fase di evaluation dalla conversione stessa.

**`PDF_JUDGE_PROMPT`:** istruisce il giudice a valutare solo la *fedeltà* (ogni elemento del Markdown è presente nell'originale?) e non la qualità Markdown in generale — evita bias verso stili di formattazione.

---

## `dependency_parsing/dependency_parsing.py` — Annotazione linguistica

`DependencyParser` applica analisi delle dipendenze morfosintattiche ai file Markdown convertiti.

**Flusso:**
1. Markdown → plain text (via `markdown` + BeautifulSoup per rimuovere il markup)
2. Stanza pipeline per lingua (`it`, `de`): tokenize + MWT + POS + lemma + depparse
3. Output in CoNLL-U (standard NLP) e/o JSON

**Lazy loading:** le pipeline Stanza vengono caricate solo al primo `run()`, non nell'`__init__`. Se il modello non è già scaricato, lo scarica automaticamente.

**Multi-lingua:** la stessa pipeline elabora tutti i file con tutte le lingue configurate. Il suffisso `_{lang}` viene aggiunto al nome del file di output solo se ci sono più lingue attive (per non rompere i path quando si usa una sola lingua).

**Uso nel progetto:** il parsing è l'ultimo stadio della pipeline — produce i dati linguistici strutturati (CoNLL-U) che possono essere usati per analisi di corpus, addestramento di modelli NLP o ricerca linguistica sul corpus di testi in italiano e tedesco.

---

## Decisioni architetturali principali

| Decisione | Alternativa scartata | Motivazione |
|-----------|---------------------|-------------|
| Batching vLLM per libro intero | Pagine una ad una | Throughput molto superiore grazie al continuous batching |
| Rasterizzazione PDF in PNG | Estrazione testuale PyMuPDF | Preserva layout visivo, formule, tabelle complesse |
| Chunking EPUB per tag HTML | Finestre di testo fisse | Chunk semanticamente coesi, nessun testo spezzato |
| Prompt di sistema costanti | Prompt con variabili inline | Massimizza prefix cache hit rate con APC vLLM |
| Eval pages salvate durante conversione | Rileggere i file originali per eval | Disaccoppia le fasi; eval funziona senza i file originali |
| Resume basato su esistenza del `.md` | Flag in database/JSON | Zero overhead, funziona anche dopo crash o interruzioni |
| Due LLM separati (VL + testo) | Un solo modello multimodale | Ottimizzazione: il modello testo è più veloce e usa meno VRAM per EPUB/metadati |

---

## Test

I test usano `tmp_path` di pytest per l'isolamento completo (nessun file di stato globale). I modelli LLM sono sostituiti da stub (`FakeLLM`) che restituiscono JSON fissi, quindi i test girano senza GPU.

- **`test_book_converter.py`:** testa il resume (`_already_converted`) e il corretto skip per tutti e tre i metodi run
- **`test_metadata.py`:** testa `_parse_json` (parsing robusto), `collect_samples` (selezione pagine) e `run` (logica CSV append/skip)
- **`test_utils.py`:** testa `sample_indices` (garanzie sui primi 10 indici, stratificazione) e `pil_to_data_url` (round-trip PNG)
