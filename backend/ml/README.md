# Sketch2Vision ML foundation

This directory prepares reproducible data for the academic Pix2Pix Conditional
GAN requirement. It does not contain a trained model, inference endpoint, or
weights. The architecture can later expose one sketch-to-image inference
interface backed by the trained Pix2Pix model or, if needed, a general
pretrained sketch-conditioned model.

## Intended stages

1. Obtain and document a diverse collection of source images whose terms allow
   the intended research and machine-learning use.
2. Split source images into train, validation, and test sets before deriving
   edge maps. The deterministic manifest records stable source identifiers.
3. Derive an edge map from each source image and retain the corresponding
   transformed source image as the target. Keep pair geometry identical; every
   resize, pad, crop, or other geometric augmentation must be shared by the pair.
4. Train/evaluate Pix2Pix with training pairs, use validation data for model
   selection, and reserve test data for final evaluation.
5. Later add inference behind the app's model interface. No model is connected
   by this preparation step.

Automatically generated Canny edges are not the same as human freehand
sketches. Evaluate generalization to actual user sketches separately. Diverse
edge/image pairs may improve object coverage but cannot guarantee recognition
of every possible sketch.

## Current preparation CLI review

`scripts/prepare_dataset.py prepare` accepts an input directory and a separate
output directory. It recursively scans the input for `.jpg`, `.jpeg`, and
`.png` files; other formats are ignored. A simple input layout can be:

```text
source-images/
  bicycle/image-001.jpg
  chair/image-002.png
  dog/image-003.jpeg
```

The directory names are not interpreted as categories. The script assigns
source files to train/validation/test with default ratios 80/10/10 and a
deterministic seed (42 by default). It shuffles source identifiers before image
decoding and edge generation, so each source is assigned to at most one split.
Split sizes are approximate for small datasets. The split is not category
stratified; the script does not read labels, source metadata, licenses, or a
curated split column. Unreadable files are skipped and recorded in the output
manifest. A skipped file can leave a small split empty.

Each identifier combines the sanitized filename stem with a short hash of its
relative input path. The output has `splits.json` and `train/`, `validation/`,
and `test/` folders, each with `edges/<id>.png` and `targets/<id>.png`. Edges
are grayscale Canny maps; targets are the corresponding color source images
saved as PNG. `--image-size WIDTH HEIGHT` defaults to 256 by 256.

**Aspect-ratio handling:** `--resize-mode stretch|pad` controls resize behavior
and defaults to `stretch` for backward compatibility. `stretch` retains the
legacy direct resize and can deform objects. Opt-in `pad` applies EXIF
orientation before processing, uniformly fits the image within the requested
dimensions, centers the RGB target content on white, and places the edge map on
black at the exact same offset. Edges are extracted from the resized content
before adding padding, avoiding an artificial outline at the padded border.
The output `splits.json` records the mode, requested dimensions, pad colors,
and each successfully processed source's oriented size, resized size, and
padding amounts.

## Candidate source comparison

The facts below are from the linked dataset or policy pages checked for this
plan. “Still confirm” items are not approvals: read the exact image-level
license and source terms before using any selected file. Dataset annotations,
images, and code can have different rights.

### Open Images V7 — recommended provisional pilot

- **Coverage and size (verified):** about 9.18 million images overall, with
  image-level labels spanning 20,638 classes; a 1.9-million-image portion has
  denser annotations. This is broad object coverage for an edge-to-image
  academic experiment.
- **Download and subset (verified):** the official page offers a downloader
  driven by a text list of image IDs and split names, so a selected subset can
  be fetched without taking the full archives. It also documents class and
  `max_samples` subset selection through FiftyOne. The image metadata includes
  image ID, original and landing-page URLs, license, author, title, original
  size, and rotation.
- **Rights (verified / still confirm):** per-image license metadata is
  available, but the dataset maintainers explicitly say they do not warrant
  each image's license identification and ask users to verify each image.
  Preserve the landing page, author, title, license URL, image ID, rotation,
  and reported original size for every selected source. Attribution is required
  when the individual license requires it. Verify each license and source
  platform terms for local academic training, derived edge/target storage,
  publication, and any later model release. Do not put source images or derived
  pairs in public GitHub unless redistribution and derivative rights are
  confirmed.
- **Fit:** strongest category labels and manageable ID-based subset workflow
  for the proposed pilot, conditional on per-image review. The official source
  page also offers reduced-resolution thumbnails, but their size/contents can
  change; the pilot should record whether originals or thumbnails were used.

Official references: [Open Images V7 overview and download instructions](https://storage.googleapis.com/openimages/web/download_v7.html),
[image metadata fields and per-image license caveat](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

### MS COCO 2017 — useful benchmark, not the first download

- **Coverage and size (verified):** COCO is a scene dataset with 80 object
  categories. The 2017 labeled train and validation splits contain 118,287 and
  5,000 images; the larger collection is commonly described as about 330,000
  images.
- **Download and subset (verified):** the official site distributes image
  archives and annotations. Train and validation archives are commonly listed
  at roughly 18 GB and 1 GB, respectively, before extraction. A partial subset
  is possible through tools such as FiftyOne or by selecting image IDs and
  downloading individual source URLs; the normal official archive route is not
  a small selective download.
- **Rights (verified / still confirm):** COCO's annotations and website are
  CC BY 4.0, but the consortium says it does not own image copyrights and that
  image use must follow Flickr's terms. The COCO records include image-level
  license metadata. Review each image's actual source/license and Flickr terms;
  annotation licensing does not grant image rights. Do not redistribute image
  files, derived pairs, or trained weights publicly without checking the
  applicable rights.
- **Fit:** good academic scene/object benchmark, but its full train/validation
  download can be a poor fit once archives,
  extraction, and paired PNG outputs are considered. A small selected subset
  could fit, but requires a careful image-level rights and download workflow.

References: [COCO overview/download/terms](https://cocodataset.org/),
[COCO terms of use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm),
[COCO 2017 subset options](https://docs.voxel51.com/dataset_zoo/datasets/coco_2017.html).

### Wikimedia Commons — flexible curated alternative

- **Coverage and size:** Commons is a large media repository, not one fixed
  object dataset with a defined image count or category balance. A project can
  build a small, curated collection across chosen object categories.
- **Download and subset (verified):** use category searches and the MediaWiki
  API to find files and retrieve file information, then download only selected
  items. The API does not make every license/author field reliably structured;
  individual file-description pages remain important.
- **Rights (verified / still confirm):** Commons accepts freely licensed or
  public-domain media, and each file page states its license. Attribution and
  source credit may be required; ShareAlike terms can constrain redistribution
  of derivatives. Verify file-page provenance, author, license version,
  attribution wording, and any non-copyright restrictions. The platform's
  acceptance of a file is not a project-specific legal review of ML training
  or model release.
- **Fit:** can be appropriate for an academically curated pilot, especially if
  the team can audit every file, but it requires more manual curation and
  provides no guaranteed category balance or unified image split. Public
  GitHub data release would require per-file compliance and attribution.

References: [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/en),
[Commons licensing policy](https://commons.wikimedia.org/wiki/Commons:Licensing),
[machine-readable file metadata and its limits](https://commons.wikimedia.org/wiki/Help:Machine-readable_data),
[Commons API license data](https://commons.wikimedia.org/wiki/Commons:Commons_API).

### Smithsonian Open Access metadata-only feasibility adapter

`scripts/build_smithsonian_manifest.py` queries the official Smithsonian Open
Access Search API for up to 50 records per selected category (Person, Dog,
Cat, Car, Bicycle, Chair, Bird, Flower, Tree, Building). It requests metadata
only. It may record `online_media` URLs, but never opens them or downloads image
bytes. Search matches are candidate leads: the script marks category relevance
as `unverified` and does not claim that every result depicts the requested
subject.

The adapter writes three outputs:

- an eligible-candidate CSV containing records whose rights/access field is
  explicitly `CC0` (or its canonical CC0 1.0 marker);
- a separate review CSV containing missing, unknown, non-CC0, or otherwise
  uncertain rights records, plus records without a stable item ID;
- a JSON summary with per-category search counts, unique IDs, explicit-CC0
  counts, uncertain-rights counts, shortfalls, duplicate IDs, and missing
  metadata counts.

Both CSVs preserve the shared provenance fields and Smithsonian item metadata.
Every record remains `pending_manual_review`, including rows in the explicit
CC0 view. CC0 inclusion is a discovery filter, not approval; check the current
item page and any third-party privacy, publicity, trademark, cultural, donor,
or other restrictions before acquisition or reuse. Missing data stays blank or
`unknown`; category relevance must also be checked by a person. The API key is
read from `SMITHSONIAN_API_KEY` and is not written to output files. Do not put
the key in command arguments or commit it.

Obtain an API key through [api.data.gov](https://api.data.gov/signup/). In
Windows PowerShell, set it for the current terminal session and run from the
repository root:

```powershell
$env:SMITHSONIAN_API_KEY = Read-Host "Smithsonian API key"
New-Item -ItemType Directory -Force backend\ml\metadata | Out-Null
.\backend\.venv\Scripts\python.exe backend\ml\scripts\build_smithsonian_manifest.py `
  --eligible-csv backend\ml\metadata\smithsonian-eligible.csv `
  --review-csv backend\ml\metadata\smithsonian-review.csv `
  --summary-json backend\ml\metadata\smithsonian-summary.json
```

To run a smaller search, repeat `--category` and lower the cap (never above
50):

```powershell
.\backend\.venv\Scripts\python.exe backend\ml\scripts\build_smithsonian_manifest.py `
  --eligible-csv backend\ml\metadata\smithsonian-dog-eligible.csv `
  --review-csv backend\ml\metadata\smithsonian-dog-review.csv `
  --summary-json backend\ml\metadata\smithsonian-dog-summary.json `
  --category Dog --max-candidates-per-category 10
```

Existing outputs are protected; use `--overwrite` only when replacement is
intentional. The API's `rows`/`start` pagination is used as needed. Deduplication
uses the stable Smithsonian item ID; records with no ID are retained in the
review CSV and counted as incomplete rather than silently discarded. Image
URLs are provenance only. The included
`metadata/smithsonian_manifest_sample_summary.json` documents the summary
shape and explicitly has null counts because it is not a live API result.

The client uses the Smithsonian response envelope: HTTP success is separate
from the JSON application field (`responseCode` 1 is success), and search rows
are at `response.rows`. On failure, the CLI reports the HTTP status and a
shortened response excerpt with API-key values redacted. A `responseCode` 0 or
other non-success value is an API-level failure even when HTTP status is 200;
check that the key is valid and that the reported message is safe to share.

The Smithsonian [Open Access FAQ](https://www.si.edu/openaccess/faq) documents
CC0 designation, third-party-rights limitations, media formats, attribution
guidance, and API access. The official
[Open Access API repository](https://github.com/Smithsonian/smithsonian-openaccess)
documents the API client and the api.data.gov key requirement. The README's
research comparison is in [`dataset_source_research.md`](dataset_source_research.md).

Run the offline mocked tests and CLI help without network access:

```powershell
\.\backend\.venv\Scripts\python.exe -m unittest backend.ml.tests.test_smithsonian_manifest -v
\.\backend\.venv\Scripts\python.exe backend\ml\scripts\build_smithsonian_manifest.py --help
```

#### Metadata-only Commons manifest adapter

`scripts/build_commons_manifest.py` uses the official Commons Action API to
search the File namespace, paginate search results, then request file
`imageinfo` and `extmetadata` in small batches. It writes metadata to a CSV and
does not fetch `source_url` or any image bytes. The default cap is 30 selected
JPEG/PNG files per category (maximum one category selection per unique Commons
page ID and SHA-1); it can return fewer and reports the selected count and
shortfall rather than claiming the requested quota was achieved. Search results
may span categories: one output row is kept per Commons file and
`all_selected_categories` records every query that found it, while
`selected_category` records the first query that selected the row.

Each category query requests photographs and excludes common artwork terms
(such as posters, diagrams, illustrations, logos, paintings, and icons) in the
search expression. Returned metadata is also screened for these explicit
signals, and results with photo-related metadata are preferred. This is a
metadata heuristic: Commons titles, descriptions, and categories can be
incomplete or misleading, so candidates still need visual review after any
separately approved acquisition. The adapter also uses creator plus normalized
title tokens to skip likely repeated views of the same subject when that is
detectable. It does not infer duplicates from license labels.

`--max-per-creator-per-category` is a **soft** diversity limit (default 3).
Candidates above the limit are deferred while alternative creators are chosen;
deferred candidates fill any remaining quota so creator concentration alone
does not create a category shortfall. Set the value to `0` to disable this
preference. The summary records skipped artwork signals, likely duplicate
subjects, creator counts, and any cap fallbacks.

Run from the repository root. Set `SKETCH2VISION_CONTACT` to a real project
contact before making requests; if `--user-agent` is omitted, the CLI composes
it into the identifying User-Agent. The contact value is not printed:

```powershell
New-Item -ItemType Directory -Force backend\ml\metadata | Out-Null
$env:SKETCH2VISION_CONTACT = Read-Host "Project contact for the Commons User-Agent"
.\backend\.venv\Scripts\python.exe backend\ml\scripts\build_commons_manifest.py `
  --output-csv backend\ml\metadata\commons-candidates.csv `
  --max-candidates-per-category 5 `
  --max-per-creator-per-category 2
```

Run only selected categories by repeating `--category`, for example:

```powershell
.\backend\.venv\Scripts\python.exe backend\ml\scripts\build_commons_manifest.py `
  --output-csv backend\ml\metadata\commons-dogs.csv `
  --category Dog --category Cat --max-candidates-per-category 5 `
  --max-per-creator-per-category 2
```

The supported categories are Person, Dog, Cat, Car, Bicycle, Chair, Bird,
Flower, Tree, and Building. The summary defaults to a sibling file named
`<manifest-stem>.summary.json`; `--summary-json` can set its path. Use `--help`
for all options. Existing output is
protected; pass `--overwrite` only when replacement is intentional. The client
uses one request at a time, a configurable timeout, bounded retries, and the
server's `Retry-After` value for HTTP 429/503 responses. If that header is
absent or invalid, it waits five seconds before retrying. It does not classify
unfamiliar license templates. Missing creator, license, attribution, and file
properties remain `unknown` (with raw API data and metadata-status notes where
available). All rows are `pending_manual_review` regardless of the API values.

The CSV carries the shared provenance fields (`source_identifier`,
`source_path`, source URLs, creator, license/attribution fields, raw metadata,
and review status) plus Commons page ID, file title/timestamp, MIME type,
dimensions, byte size, and SHA-1. Once a file is separately acquired and
manually approved, save it using its manifest `source_path` (for example,
`Dog/commons-123.jpg`) so `prepare_dataset.py --source-metadata-csv` can map it
to a prepared pair. This adapter does not grant download, training, or
redistribution permission. Review the actual file-description page and
license evidence for every record; keep status pending until that review is
documented. Commons metadata can be absent or inconsistent, so a successful
manifest build is neither a license approval nor proof that a candidate depicts
the selected category.

Automated tests use mocked Action API JSON only and never access the network:

```powershell
.\backend\.venv\Scripts\python.exe -m unittest backend.ml.tests.test_commons_manifest -v
```

### PASCAL VOC 2012 — compact but rights unresolved

- **Coverage and size (verified):** approximately 11,500 images across 20
  everyday object classes, including people, animals, vehicles, and indoor
  objects. The official training/validation archive is listed at about 2 GB.
- **Download and subset (verified):** the official challenge page offers a
  tar archive of the training/validation data and a separate development kit.
  The official page reviewed does not describe a selective per-image download;
  expect to obtain the archive before selecting a small subset.
- **Rights (still confirm):** the official page reviewed describes the
  challenge and download but does not make a clear image-reuse grant visible.
  The image owners/source terms and permission for ML training and public
  release must be resolved before use. Do not infer image rights from the
  benchmark's academic popularity or available mirrors.
- **Fit:** size and category scope suit coursework, but it is not the
  recommendation until image permissions are clear. Public GitHub release is
  not approved.

Reference: [official PASCAL VOC 2012 page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/).

### Not selected: ADE20K

ADE20K is a scene-parsing dataset, but its image terms are for non-commercial
research and education, and MIT CSAIL says it does not own the image copyrights.
It is not a general commercial-use source and is not selected for this pilot.
See the [official image terms](https://ade20k.csail.mit.edu/terms/).

## Provisional pilot and next gate

**Recommended plan, not dataset approval:** prepare a 300-image Open Images V7
pilot with ten common categories (for example: person, bicycle, car, bird, cat,
dog, bottle, chair, clock, and houseplant), targeting 30 distinct images per
category.
Deduplicate by Open Images ID before downloading; an image can have more than
one label, so record all labels and one pilot “primary category” for balance.
Use the current deterministic source-level split with the chosen seed, then
report per-split category counts; the current script does not stratify by class.
The pilot would produce roughly 240/30/30 images across train/validation/test
before unreadable files are skipped.

Check current free space immediately before any future download; the reading is
time-sensitive and is not authorization. Before download, use selected
records' `OriginalSize` values to estimate source size against a proposed
**2 GB source-download cap**. At the current 256 ×
256 output size, 300 RGB targets plus grayscale edge maps represent about 75
MiB of raw pixel data before PNG compression and filesystem overhead; actual
source and output file sizes depend on image content and encoding. This makes a
small pilot plausible in the current free space, but the exact source total is
unknown until a license-screened ID list exists. Do not download the full Open
Images collection.

Before any download, the next action is to build and review a **curated
metadata-only manifest** for the 300 proposed IDs. It should record category
labels, image ID/split, original and landing-page URLs, title, author,
license URL/version, attribution text, rotation, original byte size, and a
content hash once downloaded. Confirm the applicable rights for academic
training and generated pairs, and separately decide what could appear in a
public GitHub repository or model release. Also resolve two pipeline gaps
before processing that list: preserve the source metadata/rotation in a
sidecar. The opt-in `pad` resize mode is now available, but does not resolve
dataset licensing or access. No candidate has been approved and no download is
authorized by this plan.

Open Images' synthetic edge maps are generated from the source photos; they are
not real human freehand sketches paired with those photos. A model trained on
these pairs may learn edge-to-photo mappings but can struggle with sparse,
stylized, or ambiguous user drawings. Evaluate real freehand sketches
separately.

## Open Images V7 metadata-only pilot manifest

The proposed pilot is **300 unique candidate IDs**: 30 each for Person,
Bicycle, Car, Bird, Cat, Dog, Bottle, Chair, Clock, and Houseplant. These are
selection targets, not a populated or approved dataset. No candidate rows are
included until the official metadata CSVs have been obtained separately. The
generator below reads local CSVs only: it makes no network requests and never
fetches image bytes.

### Metadata inputs and verified schema

The official [V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html)
links to human-verified image-level labels and class descriptions. It documents
label columns `ImageID,Source,LabelName,Confidence`; use only positive
`Source=verification`, `Confidence=1` records for this pilot. The same page
documents the image-information columns `ImageID,Subset,OriginalURL,
OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,
OriginalMD5,Thumbnail300KURL,Rotation`. It says `OriginalSize` is the original
download size (the schema text does not state the unit; the generator labels a
byte interpretation as an assumption) and `Rotation` is one of 0/90/180/270; `nan` means
unavailable. The schema does **not** provide original pixel width or height.

The page's V7 “Train Image IDs” link resolves to
`https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv`.
This is an official link placed in the V7 train image-ID section, so the
documentation identifies this V6-path file as the train image metadata source
used from the V7 download page; its filename/path alone would not establish
that provenance. The V7 documentation's schema and image-ID convention also
match the information required here. However, we have not downloaded the CSV
to compare actual IDs row by row, so content-level overlap is not verified.
The V7 class-name
CSV link is `https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv`,
and the V7 human-verified train labels link is
`https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv`.
These are metadata files, not image archives, but they must still be fetched
separately by the operator. The script does not fetch them.

The V7 documentation gives a schema example but no original width/height or
authoritative attribution statement field. Width and height remain blank;
rotation is blank when absent or invalid. Missing size, URL, author, title,
landing page, author profile, subset, or license fields are flagged in
`metadata_status`. The generator tolerates missing optional image-info columns
and records those blanks; the label and class-mapping join columns remain
required. It records the source-provided author/title/license components but
leaves `attribution_text` blank rather than inventing official credit wording.
A license URL is retained verbatim. A CC name/version is parsed only when the
URL matches a Creative Commons URL. Unrecognized or missing licenses remain
unresolved. Every row is marked `pending_manual_review`.

Open Images says image-level license information is supplied but explicitly
disclaims a warranty that the per-image license is correct and asks users to
verify it. The docs identify images as CC BY 2.0 and annotations as CC BY 4.0;
the annotation license does not establish image rights. Before using any row,
review the actual image landing page and author/license evidence, required
credit wording, source-site terms, academic ML training rights, derived edge
and target storage, and any public release of images, pairs, or model weights.
Do not treat this manifest as licensing approval.

### Generate and review

After obtaining the CSV metadata separately, put it outside tracked dataset
folders (for example `C:\data\open-images-metadata\`). Use the actual
image-information CSV path that has been checked for provenance. The example
uses the current official links for labels and classes and the V6-path image
information file linked from the V7 page; stop to resolve that version caveat
before relying on it.

```powershell
cd C:\Users\Admin\OneDrive\Documents\Sketch2Vision
python backend\ml\scripts\build_open_images_manifest.py `
  --labels-csv C:\data\open-images-metadata\oidv7-train-annotations-human-imagelabels.csv `
  --image-info-csv C:\data\open-images-metadata\oidv6-train-images-with-labels-with-rotation.csv `
  --class-descriptions-csv C:\data\open-images-metadata\oidv7-class-descriptions.csv `
  --output-csv C:\data\open-images-metadata\sketch2vision-candidates.csv `
  --summary-json C:\data\open-images-metadata\sketch2vision-candidates-summary.json `
  --category Person `
  --category Bicycle `
  --category Car `
  --category Bird `
  --category Cat `
  --category Dog `
  --category Bottle `
  --category Chair `
  --category Clock `
  --category Houseplant
```

The command requires exactly ten distinct display names as found in the class
description file. It uses deterministic seeded selection (default seed 42),
assigns a multi-label image to only one primary category, and retains all
matching pilot categories in `all_selected_categories`. It rejects malformed
IDs, repeated positive label rows for the same image/category, and duplicate
image-info records for candidate IDs. Repeated `ImageID` values in labels are
otherwise expected because an image can have multiple category labels.
Selection uses a seeded augmenting-path assignment with a per-category quota,
so shared images can be reassigned to avoid a shortfall caused only by category
order. For all ten categories the summary reports unique eligible IDs with
image-info joins, selected IDs, and shortfall. Candidates with missing
image-info joins are excluded from eligibility and counted separately; the
actual selected total and whether all requested slots were filled are explicit.
It never fabricates candidates.
The CSV contains source URLs, license/author fields, reported `OriginalSize`,
rotation, blank dimension and attribution-text fields, metadata source, and
review status. Attribution wording is not generated from component metadata.
The JSON summary reports category counts and the sum of known `OriginalSize`
values. That sum is only a metadata estimate: missing sizes make it incomplete,
the official schema text does not state units, and source sizes may have
changed. The CSV calls this field `original_size_reported`; the summary compares
its numeric total to the 2 GB (2,000,000,000 byte) proposal only under an
explicit byte assumption. The comparison never clears the budget. Confirm
units and current source sizes before download.

### Current metadata availability gate

Run the local-only validator at any time; it never connects to the network.
The default location is `C:\data\open-images-metadata\`:

```powershell
cd C:\Users\Admin\OneDrive\Documents\Sketch2Vision
python backend\ml\scripts\validate_open_images_metadata.py
```

For a different folder or a temporary test fixture, pass `--metadata-dir
<path>`. Missing inputs are listed with their official URLs. Fatal schema
errors include malformed/blank image IDs, duplicate image-info IDs, and any
label ID with no image-info join. Repeated label IDs across different labels
are expected for multi-label data; exact repeated positive image/category rows
are rejected by the manifest builder. Optionally pass
`--selected-ids-csv <candidate-manifest.csv>` to require each candidate ID to
be well-formed, unique, and present in image-info metadata. The validator uses
a temporary SQLite database in the metadata directory and removes it at exit.
It reads no image content, leaves source CSVs unchanged, reports raw
`OriginalSize` samples without inferring units or summing a storage estimate,
does not create a manifest, and reports no license approval. A missing exact
attribution statement remains unfilled.

The command can be exercised safely with short synthetic CSVs in a temporary
directory using `--metadata-dir`; synthetic output only validates the local
workflow and must never be reported as Open Images records.

The required metadata inputs are the V7 human-verified train-label CSV, the V7
class-description CSV, and the image-info CSV linked as “Train” under V7 Image
IDs (the V6-path object above). The official docs provide exact URLs and
schemas but do not list object sizes. A header-only request to the three URLs
could not resolve `storage.googleapis.com` in this environment, so their
content lengths are unknown. No metadata file or image was fetched; therefore
no populated candidate manifest or per-category counts are available yet.
The three expected filenames are:

```text
oidv7-train-annotations-human-imagelabels.csv
oidv7-class-descriptions.csv
oidv6-train-images-with-labels-with-rotation.csv
```

To retrieve them later, open the [official V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html)
and use **Download image labels over 9M images → Human-verified labels →
Train**, **Metadata → Class Names (all sets)**, and **Image IDs → Train**.
Alternatively, open the three direct URLs above. Before saving, record the
server-reported size; if the browser starts transferring the file before its
size is visible, cancel and obtain the size from a trusted header/metadata
view first. Save only to `C:\data\open-images-metadata\` with the exact
filenames listed above. The V6 path is the link the V7 page provides, but its
row-level ID overlap with V7 labels still needs a local comparison. In this
run, the explicitly relevant project locations
(`backend/ml`, `backend/ml/datasets`, and `C:\data\open-images-metadata`) had
no matching metadata CSVs. Header-only requests to all three official URLs
failed with DNS resolution errors, so sizes could not be obtained. No metadata
file or image was fetched and no real manifest/counts were generated. Stop
until the CSV sizes and row-level compatibility can be checked from a working
network location. After that check, fetch only the three metadata files if
their total is acceptable. Do not download image bytes until the resulting
manifest has been manually license-reviewed and the source-size total is
confirmed under the 2 GB budget.

The script refuses to overwrite either output unless `--overwrite` is passed.
Review every CSV row manually, resolve missing fields and the image-info
version/provenance caveat, and record decisions before preparing an image ID
list for download. The official downloader accepts a selected text list of
`split/ImageID` entries, allowing a subset rather than a full archive; that
download step must wait until license review and the 2 GB source-size check
are complete. The current disk-space reading is time-sensitive; check again
before a future download and do not treat free space as approval.

## Folder format

The output directory is structured as follows:

```text
dataset/
  splits.json
  train/
    edges/<stable-id>.png
    targets/<stable-id>.png
  validation/
    edges/<stable-id>.png
    targets/<stable-id>.png
  test/
    edges/<stable-id>.png
    targets/<stable-id>.png
```

Identifiers are derived deterministically from each source's relative path.
Targets and edge maps are stored as PNG at the configured dimensions (default
256 by 256). Source images are resized once before edge extraction, so both
members of each pair share the exact same geometry. Splits are assigned before
decoding, resizing, or generating pairs so a source cannot leak between splits.
Small collections can have approximate rather than exact ratios; with at least
three images, the script keeps all three splits represented.

## Setup and preparation on Windows

Use the existing backend environment and install the image-processing library
only when preparing data. This is separate from `backend/requirements.txt` and
does not install PyTorch:

```powershell
cd C:\path\to\Sketch2Vision
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\ml\requirements-optional.txt
python backend\ml\scripts\prepare_dataset.py prepare `
  --input-dir C:\data\sketch2vision-sources `
  --output-dir backend\ml\datasets\experiment-01 `
  --seed 42 `
  --image-size 256 256 `
  --resize-mode pad `
  --max-images 300
```

The input directory must already contain local `.jpg`, `.jpeg`, or `.png`
images. The `--image-size WIDTH HEIGHT` option sets the paired output
resolution; use a smaller value such as `32 32` for a quick synthetic check.
`--resize-mode pad` is opt-in: it corrects EXIF orientation, preserves source
aspect ratio, places target pixels over white padding, and places matching
edge pixels over black padding. Omitting the option preserves existing
`stretch` behavior. No automatic downloads occur. Non-empty outputs are
protected; pass `--overwrite` only when replacing that output is intended.

## Validation

```powershell
python backend\ml\scripts\prepare_dataset.py validate `
  --output-dir backend\ml\datasets\experiment-01
```

Validation checks matching edge/target filenames, readable images, equal pair
dimensions, all expected pair files, and train/validation/test source ID
disjointness. Preparation and validation report per-split counts plus valid,
skipped, and invalid totals. Unreadable source files are skipped and listed in
`splits.json`; validation exits nonzero when pair or split integrity checks
fail.

### Strict training preflight validation

Before training, run the read-only strict validator against the prepared data:

```powershell
python backend\ml\scripts\validate_training_dataset.py `
  --dataset-dir backend\ml\datasets\experiment-01
```

It requires `splits.json` to explicitly declare positive
`image_size.width`/`height`, `preprocessing.resize_mode` (`stretch` or `pad`),
and `source_ids` lists for all three splits. It checks manifest-to-disk IDs,
one-to-one edge/target pairing in both directions, split leakage, file
readability, declared dimensions, one-channel edge maps, and three-channel
color targets. OpenCV decodes target pixels as BGR; the channel check confirms
a three-channel color target without altering color data. The optional
`--expected-size WIDTH HEIGHT` also requires the manifest and files to match
an explicitly supplied size. Missing manifest dimensions are always an error
and are never silently replaced with `config.py` defaults.

For `pad` mode, the manifest must identify centered padding, white target
padding, and black edge padding. Optional category summaries are read from
`source_metadata`, `category_by_source_id`, or `categories` metadata. Optional
per-source provenance is read from `source_metadata`; missing fields and
unapproved review statuses are reported as warnings. `pending_manual_review`
remains unapproved. Warnings do not fail structural validation; any fatal
structural error returns a nonzero exit code. The command only reads dataset
files and does not rewrite or delete them.

For example, require 256 × 256 explicitly with:

```powershell
python backend\ml\scripts\validate_training_dataset.py `
  --dataset-dir backend\ml\datasets\experiment-01 `
  --expected-size 256 256
```

Passing this check establishes structural consistency only. It does not
approve image licenses, confirm dataset terms, or measure model quality.
Keep licensing review and model evaluation as separate gates.

## Source provenance and license review

`prepare_dataset.py prepare` now writes one versioned `source_metadata` record
for every successfully prepared pair, keyed by the generated pair `source_id`.
Real data is the default classification. Use `--source-metadata-csv` to pass a
local candidate/review CSV; no services are contacted. The metadata CSV maps a
row by exact relative `source_path` (recommended) or by an external identifier
(`source_identifier`, `image_id`, or `source_id`) matching the source filename
stem. For Open Images candidate manifests, retain the `ImageID` as the local
filename stem or create an explicit `source_path` mapping column. Duplicate or
ambiguous matches fail instead of attaching metadata to the wrong image.

The `splits.json` record schema is:

```json
{
  "dataset_kind": "real",
  "provenance_schema_version": 1,
  "source_metadata": {
    "prepared-pair-id": {
      "source_id": "prepared-pair-id",
      "source_identifier": "external-image-id",
      "source_path": "category/image.jpg",
      "source_url": "https://...",
      "source_page_url": "https://...",
      "license_name": "CC BY",
      "license_identifier": "https://creativecommons.org/licenses/by/4.0/",
      "attribution_required": "unknown",
      "attribution_text": "",
      "creator": "",
      "license_review_status": "pending_manual_review",
      "reviewer_notes": "",
      "source_metadata_url": "...",
      "selected_category": "...",
      "record_type": "real",
      "provenance_schema_version": 1,
      "provenance_issues": []
    }
  }
}
```

Absent or invalid values remain blank/`unknown` and are listed in
`provenance_issues`; the status defaults to `pending_manual_review`. The Open
Images manifest builder emits matching canonical columns while retaining its
original metadata fields. It leaves `attribution_required` unknown and the
status pending; it does not compose credit wording or infer approval from a
license URL, license name, author field, or dataset membership.

Before preparing real images, review each source record against its image or
landing page and applicable license version/terms. Confirm permission for the
intended academic training, derived edge/target files, attribution duties,
publication, and any planned redistribution separately. Record the actual
attribution requirement, verbatim/required credit or creator when applicable,
and reviewer notes identifying the reviewed evidence. Only then may a human
set `license_review_status` to `approved`; URL presence or a license label is
not sufficient. Keep the original source and metadata references for audit.

Example real-data preparation (files must already be locally available):

```powershell
python backend\ml\scripts\prepare_dataset.py prepare `
  --input-dir C:\data\reviewed-images `
  --output-dir backend\ml\datasets\reviewed-128 `
  --image-size 128 128 --resize-mode pad `
  --source-metadata-csv C:\data\reviewed-images.csv
```

Without a CSV row, preparation still writes a real-data record with blank
provenance and pending review. The training provenance gate checks every
split record, including correct pair-ID mapping, source URL/page, license name
or identifier, explicit `attribution_required` true/false, reviewer notes,
and an explicit approved status. If attribution is required, attribution text
or creator is also required. Missing evidence, pending/rejected status, or
incorrect mapping blocks training. The trainer's `--allow-synthetic-data`
option is accepted only when `splits.json` is explicitly marked synthetic and
every pair has a synthetic record; it cannot override a real or unclassified
dataset. Synthetic records are marked `not_applicable_synthetic`, never
`approved`.

For isolated generated test fixtures only:

```powershell
python backend\ml\scripts\prepare_dataset.py prepare `
  --input-dir C:\data\generated-fixtures `
  --output-dir backend\ml\datasets\synthetic-test `
  --dataset-kind synthetic --image-size 128 128 --resize-mode pad
python backend\ml\train.py --dataset-dir backend\ml\datasets\synthetic-test `
  --allow-synthetic-data --epochs 1 --base-channels 1 --device cpu
```

This test path does not confer rights on source images and cannot be used to
clear real records. Structural validation remains separate from license
approval and model-quality evaluation.

## Git and large files

`backend/.gitignore` excludes the contents of `ml/datasets/`, `ml/checkpoints/`,
and `ml/outputs/`, while retaining each `.gitkeep`. Keep scripts and
documentation tracked. Do not commit large image collections, generated pairs,
model checkpoints, run outputs, or secrets.

## Limitations

This preparation step creates a simple edge-to-photo paired dataset, not human
sketch annotations. Canny edges can omit semantic intent or emphasize texture,
and results trained on these pairs may not transfer well to sparse or stylized
user sketches. The default `stretch` resize can distort aspect ratios; use
`--resize-mode pad` to preserve them. Any geometric changes must be shared by
both pair members.

OpenCV is optional. Install `backend/ml/requirements-optional.txt` to run the
script; no training framework or dataset is installed by this foundation.

## Pix2Pix training pipeline architecture and implementation status

The U-Net, PatchGAN, loss, normalization, data loader, training, validation,
and checkpoint modules are implemented. No real-data training has been run.
The project
creates paired Canny-edge/photo PNGs and validates filename pairing, image
readability, pair dimensions, channels, and source-ID split disjointness. Its
preparation defaults are 256 × 256, seed 42, 80/10/10 splits, and Canny
thresholds 100/200. Resize defaults to legacy stretch, with opt-in
aspect-preserving padding. Source-level provenance, including source URLs,
license evidence, review status, selected category, raw metadata, and pair
mapping, is carried into `splits.json`. Candidate selection can balance the
pilot categories, but the later train/validation/test source split is
deterministic rather than category-stratified; review its category counts
before training.

Pix2Pix is appropriate for this academic goal because it learns a paired
conditional mapping using a generator, a condition-aware discriminator, and a
reconstruction objective. The proposed design follows the paired
image-translation formulation in [Isola et al., CVPR 2017](https://arxiv.org/abs/1611.07004).

### Step 4B.4 implementation specification (not implemented)

The following module boundaries and contracts are the implementation target;
this section does not add model code, a training loop, Colab notebook, or API
route.

```text
backend/ml/
  config.py                       shared data and initial model defaults
  models.py                       UNetGenerator, PatchGANDiscriminator, init (implemented)
  losses.py                       adversarial and weighted L1 losses (implemented)
  transforms.py                   uint8 <-> [-1,1] tensor conversion (implemented)
  scripts/prepare_dataset.py      paired image preparation
  scripts/validate_training_dataset.py  strict structural preflight
  scripts/smoke_test_pix2pix.py   synthetic device-aware forward/backward check
  tests/test_pix2pix_models.py    synthetic CPU model/loss tests
  dataset.py                      paired PNG/JPEG loader with shared [-1,1] contract (implemented)
  training.py                     one-epoch Pix2Pix optimization logic (implemented)
  evaluation.py                   validation L1, fingerprints, RNG and checkpoints (implemented)
  artifacts.py                    JSONL logs and validation grids (implemented)
  train.py                        train/validation/checkpoint CLI (implemented)
  inference.py                    planned standalone inference interface
  notebooks/train_colab.ipynb     planned thin Colab launcher
```

Keep model, loss, data, and inference modules free of Colab notebook imports.
The notebook should select/check the runtime, stage an already approved dataset,
set explicit configuration values, and call the shared training entry point.
Training code should run as a package/CLI locally or in Colab without copying a
second model implementation into the notebook.

#### Tensor and preprocessing contract

- Batches use NCHW `float32`: edge condition `B × 1 × H × W`, target and
  generator output `B × 3 × H × W`. Inputs and targets normalize from uint8
  `[0,255]` to `[-1,1]`; black maps to -1 and white to +1. Generator output
  uses `tanh`, hence also lies in `[-1,1]`.
- Edge files are decoded as one-channel grayscale. Targets must be three-channel
  RGB. The training loader must explicitly convert OpenCV BGR to RGB (or use an
  RGB-native decoder), then normalize. Do not silently drop alpha or expand a
  grayscale target: reject it or prepare a corrected dataset first.
- The loader consumes the already prepared files and declared dimensions; it
  must not resize each pair independently. Prepare the training set with
  `--resize-mode pad` so targets are 3-channel, orientation is corrected, and
  shared geometry is preserved. White target padding is paired with black edge
  padding at exactly the same offset. The legacy `stretch` mode remains
  supported by preparation but can preserve grayscale/alpha channels and may
  distort aspect ratio; the strict preflight validator will reject targets
  that are not three-channel.
- Use square power-of-two dimensions for the initial model (128 or 256) so
  encoder/decoder stages and skip connections align exactly. The configurable
  model depth must be checked against H and W; unsupported dimensions should
  fail before training rather than crop or misalign tensors. Keep the
  preprocessing mode, dimensions, Canny thresholds, and transform version in
  the run config/checkpoint.
- Training conditions are generated Canny maps, not user freehand sketches.
  For a later UI inference call, convert the white-background/dark-stroke
  canvas into a one-channel black-background/white-stroke condition (grayscale
  and invert, with a documented threshold/softness policy), then apply the
  configured fit/padding geometry. Do not run Canny over the user drawing.
  Stroke width and style remain a domain gap requiring separate evaluation.

#### Model and objective

- **U-Net generator:** a configurable seven-downsample U-Net at 128 × 128 and
  eight downsample stages at 256 × 256. Use 4 × 4 stride-2 padded convolutions
  in the encoder, LeakyReLU(0.2), and skip concatenations in the decoder.
  Start with base width 32, doubling channels to a cap of 512. The first and
  bottleneck stages omit normalization; use BatchNorm in intermediate stages,
  ReLU in decoder blocks, and dropout 0.5 on the three deepest decoder blocks
  as a fixed initial setting. The output layer returns three channels through
  `tanh`. The innermost 1 × 1 bottleneck omits BatchNorm so batch size 1 is
  valid.
- **Conditional 70 × 70 PatchGAN:** concatenate the one-channel condition and
  three-channel real/fake image into `B × 4 × H × W`. Use the standard
  4 × 4 convolutional patch stack (stride 2 in its first three blocks,
  stride 1 in the final blocks), channels `base/2×/4×/8×` (32/64/128/256 by
  default), LeakyReLU and
  intermediate normalization.
  Output a grid of raw `B × 1 × h × w` logits, with no sigmoid layer. Require
  spatial inputs large enough for the 70-pixel receptive field.
- **Losses:** use `BCEWithLogitsLoss` on patch logits. Let
  `L_D = 0.5 × (BCE(D(edge,target), 1) + BCE(D(edge,stopgrad(G(edge))), 0))`.
  Let `L_G = BCE(D(edge,G(edge)), 1) + 100 × mean(L1(G(edge),target))`.
  Log adversarial and L1 terms separately. Keep lambda=100 fixed for the first
  baseline; any later tuning must use validation only. `BCEWithLogitsLoss`
  combines sigmoid and binary cross entropy in a numerically stable operation
  ([PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)).
  This is the paired conditional-adversarial plus reconstruction objective
  described by the [Pix2Pix paper](https://arxiv.org/abs/1611.07004), not a
  claim of output quality.
- **Initialization and optimization:** initialize convolution weights from
  `Normal(0, 0.02)` and biases to zero; initialize BatchNorm weights from
  `Normal(1, 0.02)` and biases to zero. Use separate Adam optimizers for G and
  D with learning rate `2e-4`, betas `(0.5, 0.999)`. Start with no scheduler
  and full precision; treat AMP as a separate recorded experiment. Keep this
  initialization in tests and config; PyTorch documents the supported
  initialization operations in [`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html).
  These are fixed baseline choices, not values proven optimal for this dataset.

#### Training interface and split discipline

The training config should expose image size, batch size, epochs, base
channels, learning rate, L1 weight, seed, worker count, checkpoint interval,
output directory, device, and AMP. Initial Colab defaults: **128 × 128**,
batch size **1**, base channels **32**, **50** maximum epochs, seed **42**,
checkpoint interval **5** epochs, and AMP disabled. At each launch, inspect
the assigned GPU and available VRAM; increase batch/resolution only after a
short memory smoke check. The existing `config.py` data default remains
256 × 256, so create a matching 128 × 128 prepared dataset explicitly or
choose 256 × 256 training settings. Do not let the loader silently resize.

Train gradient updates on `train` only. Evaluate validation loss and save
validation previews after each epoch without optimizer updates. Select the
best checkpoint by lowest validation L1 (with fixed validation previews as a
sanity check); keep the test split sealed during architecture/configuration
selection. Run test metrics exactly once after the model/configuration is
frozen, and do not use test results to revise hyperparameters. For the
initial experiment, keep a fixed optimizer rate and cap at 50 epochs; stop
early only by a documented validation rule, never by test performance.

#### Logs, checkpoints, and resume contract

Write epoch/step JSONL or CSV records for G adversarial loss, G L1, D loss,
validation L1, learning rates, elapsed time, and peak allocated GPU memory.
At fixed validation IDs, save a grid with input edge, generated RGB output,
and paired target. The grid must use `eval()`/inference mode and never sample
the test split during training.

`latest` checkpoints at the configured interval and at normal completion;
save `best` whenever validation L1 improves (ties keep the earlier checkpoint).
Include model architecture/settings, G and D state dicts, both optimizer
states, completed epoch/global step, config, best metric/history, Python,
NumPy and Torch CPU/CUDA RNG states, data-loader generator state, software and
device metadata, preprocessing version, and SHA-256 fingerprint of `splits.json`.
Save atomically through a temporary sibling file and replace. Resume only from
`latest` at an epoch boundary, restoring optimizers and RNG state; refuse
changed architecture, data fingerprint, or incompatible training config
unless explicitly starting a new run. Use `best` for inference, not resume.
Store logs, grids, and checkpoints in ignored `backend/ml/outputs/` or
`backend/ml/checkpoints/`, and export from Colab only after checking dataset
terms and checkpoint contents.

#### Future inference boundary and tests

When PyTorch is available, use the core modules without any Colab imports:

```python
from backend.ml.config import PIX2PIX_BASE_CHANNELS, PIX2PIX_L1_WEIGHT
from backend.ml.losses import discriminator_loss, generator_loss
from backend.ml.models import build_pix2pix
from backend.ml.transforms import normalize_uint8_tensor

generator, discriminator = build_pix2pix(base_channels=PIX2PIX_BASE_CHANNELS)
edges = normalize_uint8_tensor(edge_batch_uint8)    # [B,1,128,128] -> [-1,1]
targets = normalize_uint8_tensor(rgb_batch_uint8)   # [B,3,128,128] -> [-1,1]
generated = generator(edges)                        # [B,3,128,128], tanh range
fake_logits = discriminator(edges, generated)
g_total, g_adv, g_l1 = generator_loss(fake_logits, generated, targets, PIX2PIX_L1_WEIGHT)
```

`UNetGenerator` defaults to `base_channels=32` and seven downsampling stages
for 128 × 128 inputs. For 256 × 256, configure eight stages. `PatchGANDiscriminator`
defaults to the same base width and accepts either `(edge_tensor, rgb_tensor)`
or a pre-concatenated four-channel tensor. It returns raw patch logits. The
`build_pix2pix` helper applies `initialize_weights`; direct class construction
uses PyTorch's layer defaults unless the caller explicitly applies that
initializer. The loss helper returns `(total, adversarial, weighted_l1)`;
for a discriminator update, run it on `generated.detach()` before calling
`discriminator_loss`.

The unit tests use only synthetic CPU tensors and do not train or access a
dataset. Run them in Colab from the repository root, with PyTorch already
available:

```python
!python -m unittest discover -s backend/ml/tests -p "test_pix2pix_models.py" -v
```

This project does not install a PyTorch build. The unit-test class reports
skips when PyTorch is absent.

#### Colab runtime smoke test

Open or clone the project into `/content/Sketch2Vision` and set the notebook
working directory to the repository root. First check the runtime; do not
assume Colab assigned a GPU:

```python
%cd /content/Sketch2Vision
import torch
print("PyTorch:", torch.__version__)
if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("CUDA device:", torch.cuda.get_device_name(0))
    print(f"GPU memory: free={free_bytes / 1024**3:.2f} GiB / total={total_bytes / 1024**3:.2f} GiB")
else:
    print("CUDA unavailable; the smoke test will use CPU")
```

Then run the standalone smoke check:

```python
!python backend/ml/scripts/smoke_test_pix2pix.py --device auto --size 128 --base-channels 32 --seed 42
```

The script reports its selected device and GPU memory when applicable, makes
synthetic inputs, checks generator and PatchGAN shapes/range, computes finite
losses, verifies discriminator and generator gradients, and by default checks
a temporary model state-dict save/load round trip. It uses backward passes to
test gradients but does not step an optimizer or train the model. The temporary
checkpoint is removed automatically. If GPU memory is insufficient, stop and
retry with an explicitly smaller `--base-channels` value; do not infer that a
training run will fit from a smoke-test pass alone.

`backend/ml/inference.py` is still planned to expose a small loader such as
`load_generator(checkpoint_path, device="auto")` returning an object with
`predict(sketch_image) -> RGB PIL.Image`. The returned object owns the loaded
generator in `eval()` mode and uses
[`torch.inference_mode()`](https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html). Keep checkpoint
architecture/preprocessing metadata self-describing; do not import Colab
notebook utilities. FastAPI can later load this object once on startup and add
an endpoint in a separate task. The inference function should accept a decoded
image rather than HTTP request objects, so it is testable without FastAPI.

Current synthetic unit tests cover generator shape/range and invalid inputs,
discriminator patch output and invalid pairs, manual GAN/L1 loss values,
gradient flow, and uint8 normalization round trips. The CPU training-pipeline
tests use temporary synthetic pairs for loader normalization, one optimizer
step, validation L1, checkpoint state round trips, resume configuration checks,
and the train/validation-only loader boundary. They skip clearly if PyTorch is
unavailable. Keep tests synthetic and small; they do not download data or run
multi-epoch training.

### Dataset contract and preparation

The future loader should consume the existing layout, with strict pair IDs:

```text
dataset/<version>/
  splits.json
  train/edges/<source-id>.png
  train/targets/<source-id>.png
  validation/edges/<source-id>.png
  validation/targets/<source-id>.png
  test/edges/<source-id>.png
  test/targets/<source-id>.png
```

The condition is a one-channel edge map (black background, white Canny edges);
the target is a three-channel RGB source photo. Both must have identical
dimensions and filename/source ID. The implemented loader normalizes both
tensors consistently to `[-1, 1]` and converts OpenCV's BGR decode to RGB.
`splits.json` stores the source-level provenance record keyed to each prepared
pair ID alongside split assignments and transforms; a separate sidecar is not
required. Only license-reviewed real items can enter training. Do not embed
source images or credentials in checkpoints or Git.

Before resizing, apply recorded orientation. The preparation script's `pad`
mode uniformly fits the image inside the requested dimensions, centers it on a
white target canvas, generates edges from resized image content, then places
those edges at the same offset on a black input canvas. This avoids stretching
objects and artificial edges along padding. Apply identical geometric
transforms to each input/target pair. Keep `stretch` as the compatibility
default unless `pad` is explicitly selected.

Create split assignments once at the source-image level, before augmentation
or edge generation, and save them in the manifest. A suitable pilot target is
300 **approved and available** distinct images across ten categories, aiming
for about 30 per category and approximately 240/30/30 train/validation/test.
Use deterministic category-aware allocation while assigning each source ID to
exactly one split; record all labels for multi-category images. If categories
cannot supply the target, report actual counts and shortfalls instead of
duplicating images. Apply only paired geometric augmentation to training data
(initially, optional horizontal flip); never augment validation or test data.
Deduplicate by stable dataset image ID and, where permitted, source checksum
before splitting.

The strict structural preflight validator now checks required files and pair
IDs, valid image decoding, one-channel edge/three-channel target shape, exact
manifest dimensions, category counts when metadata is present, source-level
split leakage, and manifest-to-file consistency. It reports licensing gaps
without approving records. Tensor normalization/ranges, content-level
duplicates, and model-specific quality checks remain outside this image-file
validator and must be checked in the future training loader/evaluation code.

### Initial experiment configuration

Use Google Colab with an explicitly selected GPU runtime for the initial
training experiment. The current Windows environment was checked as CPU-only
(Intel UHD Graphics 620, no NVIDIA CUDA tools); `backend/.venv` has no PyTorch
or torchvision. Recheck Colab's allocated GPU and free VRAM at run time rather
than assuming a fixed accelerator. Start conservatively at **128 × 128**, batch
size **1**, and generator base width **32**. Make image size, batch size, base
width, learning rate, and epoch count configuration values. Start with batch
size 1 and increase to 2 or 4 only if the allocated GPU has room. If memory is
tight, reduce resolution or model width before changing other training
settings. If the allocated GPU supports it, compare a controlled 256 × 256
run. Optional CUDA automatic mixed precision may reduce memory
use, but it must be recorded and compared with the full-precision baseline.
PyTorch documents AMP as a mixed-precision performance/memory technique; it
does not guarantee identical results across hardware or software versions
([AMP documentation](https://docs.pytorch.org/docs/stable/accelerator/amp.html)).

The proposed architecture and objective are:

- **Generator:** U-Net encoder/decoder with skip connections; one-channel edge
  input, three-channel `tanh` RGB output. Use strided 4 × 4 convolutions in the
  encoder and up-convolutions in the decoder. Record normalization and dropout
  choices in the run config; do not change them between baseline comparisons.
- **Discriminator:** 70 × 70 PatchGAN receiving the edge condition concatenated
  with either the real target or generated RGB image. Emit patch logits so the
  discriminator judges local structure while seeing the condition.
- **Loss:** conditional adversarial loss with `BCEWithLogitsLoss`, plus mean
  absolute pixel reconstruction loss. Begin with `L_G = L_GAN + 100 × L1`;
  discriminator loss is the average of real-pair and fake-pair losses. Keep
  loss terms logged separately and treat the L1 weight as a later experiment
  variable.
- **Optimizers:** Adam for both networks, learning rate `2e-4`, betas
  `(0.5, 0.999)` as the initial comparison point. Train the discriminator on
  real and detached fake pairs, then update the generator through the
  discriminator.
- **Schedule:** begin with a 50-epoch cap and validation every epoch. Save the
  final checkpoint and a best-validation checkpoint, selected by validation
  L1 with preview grids reviewed for obvious quality regressions. Use
  patience-based early stopping only as a documented safeguard against
  overfitting. The small pilot is for a pipeline demonstration, not evidence
  of broad visual quality.

### Reproducibility, checkpoints, and evaluation

Record seed 42 initially for Python, NumPy, PyTorch CPU/CUDA, the data-loader
worker seeds, and the split manifest. Record package versions, Colab GPU model,
CUDA runtime, image size, batch size, transforms, thresholds, loss weights,
optimizer settings, and dataset-manifest fingerprint. Request deterministic
PyTorch algorithms where supported and record any unsupported/nondeterministic
operations; exact repeatability is only expected under the same compatible
hardware/software stack ([PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)).

Each checkpoint should contain generator and discriminator state dictionaries,
optimizer states, epoch/step, configuration, seed/RNG state, and validation
history. Save `latest` periodically and `best` after validation selection;
write atomically so an interrupted Colab session does not leave a partial
checkpoint. Keep checkpoints, logs, and preview grids under an ignored run
directory, not in the repository. Export approved `.pt` checkpoints and their
configuration/metrics from Colab to Drive or a local download. Do not export
dataset files unless their terms permit it.

Evaluate only on the held-out test split after configuration selection.
Report per-image and aggregate RGB MAE/L1, PSNR, and SSIM at a fixed
resolution, with a fixed visual grid of input edge map, generated output, and
paired target. Include per-category counts and example failures; do not use
test metrics to select checkpoints. These metrics measure reconstruction of
the exact paired target, not user satisfaction. Generated Canny edges are
synthetic inputs derived from the target photo, not human freehand sketches.
Keep a separate real-sketch evaluation set with its own licensing/consent
review and human assessment of subject recognition, composition, and
plausibility. It has no exact paired-photo reference, so pixel metrics are not
appropriate there. The model may fail on sparse, ambiguous, stylized, or
out-of-distribution sketches; it cannot be claimed to interpret every sketch.

### Later FastAPI handoff and unresolved prerequisites

FastAPI integration is a later task: load the selected generator checkpoint
once at application startup, use the same orientation/resize/padding and
edge-map preprocessing as training, switch to evaluation/inference mode, and
run inference without gradients. Return the generated RGB image through a
separately designed endpoint. The current API only has its health route; no
model endpoint or checkpoint-loading code exists. Keep architecture code,
checkpoint config, and preprocessing version aligned between Colab and the
server, and choose CPU or CUDA device explicitly at startup.

Before training, resolve these gates:

1. Select a dataset whose individual image licenses and source terms permit the
   intended academic training, derived edge/target storage, and any planned
   public release. Open Images is not currently available locally; its
   metadata manifest is not populated, its image-info/label compatibility
   remains unverified at row level, and no image has been license-approved.
   Do not treat the proposed 300-image target as an available dataset.
2. Confirm source download size against the 2 GB proposal using documented,
   verified size units. `OriginalSize` is not yet established as a reliable
   byte count here; the budget remains unresolved.
3. Populate real provenance through a reviewed metadata CSV, manually review
   every source, and set approved statuses only when the documented evidence
   requirements are met. The trainer blocks missing, pending, rejected, or
   incomplete records; the synthetic override cannot bypass this gate.
4. Confirm the training environment's PyTorch/CUDA versions and run the
   synthetic tests and a short GPU smoke experiment before any full pilot.

No dataset or model weights are included. The model smoke test and training
implementation require a PyTorch runtime; this Windows environment did not
have PyTorch installed when checked. The training code has not been run against
a real dataset or used to train a model.

## Step 4B.7 training pipeline (implemented; training not run)

The pipeline uses the existing U-Net, conditional PatchGAN, GAN/L1 losses, and
tensor transforms. `dataset.py` loads only already prepared pairs and does no
resize or independent geometric transformation. It validates grayscale edge
maps, three-channel targets, equal pair dimensions, and expected size, converts
OpenCV BGR targets to RGB, and normalizes uint8 values to `[-1, 1]`.

`training.py` implements one epoch of updates: D sees real pairs and detached
generated pairs first; G is then recomputed with a fresh graph and optimized
through a temporarily frozen D. `evaluation.py` computes mean pixel L1 under
`torch.inference_mode()`. The CLI validates the complete train/validation/test
folder structure before any optimization, but constructs and iterates only
train and validation loaders. The test folder is never used for gradients,
metrics, previews, model selection, or tuning. Structural preflight reads the
test files only to check they are intact; final test evaluation is not part of
this implementation.

The conservative defaults are 128 × 128, batch size 1, base width 32, 50
epochs, Adam at `2e-4` with betas `(0.5, 0.999)`, L1 weight 100, seed 42,
zero workers, and checkpoints every five epochs. Resolution must be a square
power of two at least 128. For 256 × 256, the model uses eight U-Net downsample
stages. CUDA is selected only when available; otherwise CPU is used. The run
records Python, PyTorch, NumPy, OpenCV, device/GPU/CUDA information, config,
preprocessing settings, and a SHA-256 dataset fingerprint over `splits.json`
plus the train and validation pair bytes (test image bytes are excluded).
Deterministic algorithms are requested in warning mode; exact bitwise
reproducibility is not guaranteed across different hardware/software stacks.

### Colab usage

Upload or mount an already prepared dataset in Colab. Ensure the runtime has a
GPU selected and verify the runtime before training:

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM free/total: {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB")
```

From the repository root, run help and then a short, explicitly authorized
synthetic/local experiment or a fully license-reviewed dataset:

```bash
!python backend/ml/train.py --help
!python backend/ml/train.py \
  --dataset-dir /content/data/sketch2vision-128 \
  --output-dir /content/drive/MyDrive/sketch2vision/runs/pix2pix-128 \
  --image-size 128 --batch-size 1 --base-channels 32 --epochs 50 \
  --learning-rate 0.0002 --l1-weight 100 --workers 2 --seed 42 \
  --checkpoint-interval 5 --device auto
```

The dataset must already exist with `splits.json` and the strict train,
validation, and test folders. Prepare it separately at the selected size; the
trainer refuses dimension mismatches and never silently resizes. Every real
source in every split must have complete provenance and an explicit human
review status of `approved`. There is no real-data license override.
`--allow-synthetic-data` applies only to explicitly synthetic records and does
not change their status to approved. Structural validation never grants
permission to use or redistribute images.

To resume from an epoch-boundary `latest.pt`, supply the same dataset and
optimization settings, choose the same or a higher epoch ceiling, and pass the
checkpoint path:

```bash
!python backend/ml/train.py \
  --dataset-dir /content/data/sketch2vision-128 \
  --output-dir /content/drive/MyDrive/sketch2vision/runs/pix2pix-128 \
  --epochs 50 --device auto \
  --resume /content/drive/MyDrive/sketch2vision/runs/pix2pix-128/latest.pt
```

Resume restores both model and optimizer states, completed epoch/global step,
best validation score/history, Python/NumPy/Torch RNG states, and the shuffled
training loader generator state. It rejects changed dataset fingerprint,
architecture, image/batch settings, seed, optimizer settings, or worker count;
the epoch ceiling, output directory, and device may change. Checkpoints are
written atomically. `latest.pt` is saved on the configured interval and at
normal completion. `best.pt` is selected only by lower validation L1 (ties
retain the previous best). `metrics.jsonl`, `run_config.json`, and per-epoch
edge/generated/target validation grids are written beside them. The test split
does not influence any of these decisions.

### Training tests and limits

Run the synthetic tests from the project root in an environment with the
project's optional OpenCV dependency and PyTorch already available:

```bash
python -m unittest discover -s backend/ml/tests -p "test_training_pipeline.py" -v
python -m unittest discover -s backend/ml/tests -p "test_pix2pix_models.py" -v
```

The tests include a single tiny CPU optimizer step at 128 × 128 and do not
constitute a training run or a GPU-memory guarantee. No full run, real dataset
loading, Colab execution, sample-quality assessment, final test evaluation, or
real freehand-sketch generalization has been performed. The generated inputs
remain synthetic edge maps paired with their source images; this does not
establish how well the model interprets user drawings.

## Presentation-only Pix2Pix training demo (CMP Facades)

For a short Colab demonstration, use [`notebooks/Sketch2Vision_Facades_Pix2Pix_Demo.ipynb`](notebooks/Sketch2Vision_Facades_Pix2Pix_Demo.ipynb). It reuses this project's `build_pix2pix`, `train_one_epoch`, loss functions, validation L1, and checkpoint writer. The separate [`facades_demo.py`](facades_demo.py) helper downloads the approximately 29 MB official Pix2Pix Facades archive, safely extracts side-by-side paired examples, derives a one-channel Canny outline from the facade label map, and pairs that outline with the corresponding RGB facade photo.

This is a **demo dataset**, not the planned Sketch2Vision multi-category dataset. It demonstrates label-map-boundary-to-facade-photo translation; it does not demonstrate freehand sketch understanding. The quick notebook defaults to 128 × 128, batch size 1, base width 16, and a deterministic 160-train / 32-validation subset for three epochs. These are presentation settings, not an evaluated experiment. The notebook shows an initialization output, each epoch's held-out example, loss curves, inference on a test split example, and saves a demo-only `best.pt` checkpoint.

### Dataset terms and use boundary

The [CMP Facade Database page](https://cmp.felk.cvut.cz/~tylecr1/facade/) identifies the database as **CC BY-SA** and provides its dataset citation, but the page's visible license label does not name a version and says images were gathered from various sources. Attribution and ShareAlike conditions matter; image-level provenance and downstream checkpoint/model implications have not been reviewed for Sketch2Vision. The [Berkeley Pix2Pix dataset directory](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) hosts the prepared `facades.tar.gz` archive but is not a separate license grant. Review the linked terms and citation yourself before running the notebook; its acknowledgment cell is intentionally false by default. No license approval is made by this repository or by successful training.

The notebook is explicitly isolated from the strict production `train.py` provenance gate. It calls the existing low-level training epoch function for a presentation-only demonstration; this does not relax the gate, approve the images, or make the produced checkpoint suitable for Sketch2Vision's intended task. Keep the downloaded data and checkpoint in Colab's temporary directories. Do not copy the checkpoint into production checkpoint directories, publish it, or use it for the project model without the required rights review.

### Colab steps

1. Create a fresh source ZIP containing the current project, including this notebook and `backend/ml/facades_demo.py`, and upload it to Colab. Open the notebook there.
2. In Colab choose **Runtime → Change runtime type → GPU** (T4 if available). The notebook checks CUDA and stops rather than silently training on CPU.
3. Read the terms above and the notebook's dataset-terms cell. Only after deciding the limited demo use is appropriate, change `DEMO_TERMS_REVIEWED = False` to `True`.
4. Run cells in order. The first code cell asks for the current project ZIP; the data cell downloads only the Facades demo archive. No project dataset is downloaded.
5. The training cell runs three actual epochs by default. Reduce `MAX_TRAIN` or `EPOCHS` if the presentation time is short; do not describe a run as trained until that cell completes. The last cell downloads `best.pt` from the demo run.

No Colab GPU run has been performed from this repository, so runtime duration, visual quality, and checkpoint behavior on the assigned Colab hardware remain unverified. Short-run output can be poor or blurry and is not evidence of broad image generation quality.
