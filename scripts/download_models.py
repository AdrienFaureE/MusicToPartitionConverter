"""Télécharge les poids des modèles dans ``models/``.

Utilisation::

    python scripts/download_models.py                    # ByteDance + vérification de Basic Pitch
    python scripts/download_models.py --engine mt3       # checkpoints MT3 (2 x 164 Mo)
    python scripts/download_models.py --engine all
    python scripts/download_models.py --verovio-js       # verovio-toolkit-wasm.js pour web/vendor/

Les fichiers déjà présents et intacts (taille ou somme de contrôle) sont ignorés.

Sources :

- ByteDance : https://zenodo.org/record/4034264 (MD5 publié par Zenodo) ;
- MT3 : bucket public ``gs://mt3/checkpoints`` lu par l'API JSON de Google Cloud Storage (MD5 par objet) ;
- Basic Pitch : poids livrés dans le paquet pip, rien à télécharger.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BYTEDANCE_NAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
BYTEDANCE_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
BYTEDANCE_SIZE = 171_966_578
BYTEDANCE_MD5 = "22b961b77c1878239fec963362097045"

MT3_BUCKET = "mt3"
MT3_LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}&fields=items(name,size,md5Hash),nextPageToken"
MT3_OBJECT_URL = "https://storage.googleapis.com/{bucket}/{name}"

VEROVIO_JS_URL = "https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js"
USER_AGENT = "partition-ia-downloader"


def md5_of(path: Path) -> str:
    """Somme MD5 hexadécimale d'un fichier."""
    digest = hashlib.md5()  # noqa: S324 - contrôle d'intégrité, pas de sécurité
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, size: int | None = None, md5: str | None = None) -> bool:
    """Indique si un fichier existe et correspond à la taille et à la somme attendues."""
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    return md5 is None or md5_of(path) == md5


def download(url: str, destination: Path, size: int | None = None, md5: str | None = None, quiet: bool = False) -> bool:
    """Télécharge ``url`` vers ``destination``. Retourne ``False`` si le fichier était déjà valide."""
    if is_valid(destination, size, md5):
        if not quiet:
            print(f"  déjà présent : {destination.name}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:  # noqa: S310
        total = int(response.headers.get("Content-Length") or size or 0)
        done = 0
        while chunk := response.read(1 << 20):
            output.write(chunk)
            done += len(chunk)
            if not quiet and total:
                print(f"\r  {destination.name} : {done / 2**20:.0f} / {total / 2**20:.0f} Mo", end="", flush=True)
    if not quiet and total:
        print()
    if not is_valid(temporary, size, md5):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Téléchargement corrompu ou incomplet : {destination.name}")
    temporary.replace(destination)
    return True


def download_bytedance(models_dir: Path) -> None:
    """Checkpoint ByteDance (172 Mo) depuis Zenodo."""
    print("ByteDance (Zenodo, 172 Mo)")
    download(BYTEDANCE_URL, models_dir / "bytedance" / BYTEDANCE_NAME, BYTEDANCE_SIZE, BYTEDANCE_MD5)


def list_mt3_objects(prefix: str) -> list[dict[str, str]]:
    """Liste les objets du bucket public MT3 sous un préfixe."""
    items: list[dict[str, str]] = []
    token = ""
    while True:
        url = MT3_LIST_URL.format(bucket=MT3_BUCKET, prefix=urllib.parse.quote(prefix, safe=""))
        if token:
            url += f"&pageToken={urllib.parse.quote(token)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            data = json.load(response)
        items += [item for item in data.get("items", []) if not item["name"].endswith("/")]
        token = data.get("nextPageToken", "")
        if not token:
            return items


def download_mt3(models_dir: Path, model_types: tuple[str, ...] = ("mt3", "ismir2021")) -> None:
    """Checkpoints MT3 (format T5X : un dossier de nombreux petits fichiers par modèle)."""
    for model_type in model_types:
        prefix = f"checkpoints/{model_type}/"
        objects = list_mt3_objects(prefix)
        total = sum(int(item["size"]) for item in objects) / 2**20
        print(f"MT3 « {model_type} » (Google Cloud Storage, {len(objects)} fichiers, {total:.0f} Mo)")
        fetched = 0
        for index, item in enumerate(objects, start=1):
            destination = models_dir / "mt3" / item["name"]
            md5 = base64.b64decode(item["md5Hash"]).hex() if item.get("md5Hash") else None
            url = MT3_OBJECT_URL.format(bucket=MT3_BUCKET, name=urllib.parse.quote(item["name"]))
            fetched += download(url, destination, int(item["size"]), md5, quiet=True)
            print(f"\r  {index}/{len(objects)} fichiers vérifiés, {fetched} téléchargés", end="", flush=True)
        print()


def check_basic_pitch() -> None:
    """Basic Pitch embarque ses poids : on vérifie seulement leur présence."""
    print("Basic Pitch (poids inclus dans le paquet pip)")
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception as exc:
        print(f"  paquet absent ou inutilisable : {exc}")
        return
    state = "présent" if Path(ICASSP_2022_MODEL_PATH).exists() else "ABSENT"
    print(f"  modèle {state} : {ICASSP_2022_MODEL_PATH}")


def download_verovio_js(root: Path) -> None:
    """Bibliothèque JavaScript Verovio pour le rendu de secours côté navigateur (optionnel)."""
    print("Verovio JavaScript (optionnel)")
    download(VEROVIO_JS_URL, root / "web" / "vendor" / "verovio-toolkit-wasm.js")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée du script."""
    for stream in (sys.stdout, sys.stderr):  # consoles Windows en cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Télécharge les poids des modèles de partition-ia.")
    parser.add_argument("--engine", choices=("default", "bytedance", "basic_pitch", "mt3", "all"), default="default",
                        help="default = bytedance + basic_pitch (MT3 est volumineux et optionnel).")  # fmt: skip
    parser.add_argument("--models-dir", help="Dossier de destination (défaut : configuration).")
    parser.add_argument("--verovio-js", action="store_true", help="Télécharge aussi verovio-toolkit-wasm.js.")
    args = parser.parse_args(argv)

    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        from partition.config import load_settings

        models_dir = load_settings().models_dir
    print(f"Dossier des modèles : {models_dir}\n")
    failures = 0
    steps = []
    if args.engine in ("default", "bytedance", "all"):
        steps.append(lambda: download_bytedance(models_dir))
    if args.engine in ("default", "basic_pitch", "all"):
        steps.append(check_basic_pitch)
    if args.engine in ("mt3", "all"):
        steps.append(lambda: download_mt3(models_dir))
    if args.verovio_js:
        steps.append(lambda: download_verovio_js(ROOT))
    for step in steps:
        try:
            step()
        except Exception as exc:
            failures += 1
            print(f"\n  Échec : {exc}")
    print("\nTerminé." if not failures else f"\nTerminé avec {failures} échec(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
