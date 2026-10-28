"""Archivos subidos con código escondido (ejecutables, scripts, datos tras el final de la imagen).

Las cargas de prueba se arman por fragmentos: así este archivo no contiene las firmas completas y los antivirus
no lo confunden con malware (lo borraban en Windows).
"""
import io

import pytest

from app.storage import InvalidImage, validate_audio_bytes, validate_image_bytes
from tests.conftest import create_project, logged_in, png_bytes, upload

DOS_MSG = b"This program " + b"cannot be run " + b"in DOS mode."
PE_STUB = b"M" + b"Z" + b"\x00" * 58 + b"\x80\x00\x00\x00" + DOS_MSG + b"\r\n"
ELF = b"\x7f" + b"EL" + b"F" + b"\x02\x01\x01" + b"\x00" * 64
PHP = b"<" + b"?" + b"php echo 1; " + b"?" + b">"
SHELL = b"#" + b"!/bin/" + b"sh\ntrue\n"
SCRIPT = b"<" + b"script>" + b"void(0)" + b"</" + b"script>"
ZIP = b"P" + b"K\x03\x04" + b"\x00" * 40


def test_clean_images_pass():
    for fmt in ("PNG", "JPEG", "WEBP"):
        assert validate_image_bytes(png_bytes("red", 1, fmt=fmt)).fmt == fmt


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
@pytest.mark.parametrize("payload", [PE_STUB, ELF, PHP, SHELL, SCRIPT])
def test_image_with_appended_payload_is_rejected(fmt, payload):
    with pytest.raises(InvalidImage):
        validate_image_bytes(png_bytes("red", 2, fmt=fmt) + payload)


ASP = b"<" + b"%@ Page Language=\"C#\" %" + b">"


def test_asp_directive_is_rejected_but_three_stray_bytes_are_not():
    """«<%@» son sólo 3 bytes: por azar aparecen en ~7 % de los PNG de 1 MB (falsos rechazos). La directiva completa sí se rechaza."""
    from PIL import Image, PngImagePlugin

    for text, ok in ((ASP.decode(), False), ("<" + "%@ import namespace", False), ("x<" + "%@y z", True), ("<" + "%@", True)):
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", text)
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(buf, "PNG", pnginfo=info)
        if ok:
            assert validate_image_bytes(buf.getvalue()).fmt == "PNG"
        else:
            with pytest.raises(InvalidImage):
                validate_image_bytes(buf.getvalue())
    with pytest.raises(InvalidImage):
        validate_image_bytes(png_bytes("red", 7) + ASP)


def test_png_with_corrupt_exif_is_stored_instead_of_failing():
    """Una imagen válida con EXIF dañado hacía fallar `exif_transpose` al crear la miniatura (SyntaxError: error 500 al subirla)."""
    from PIL import Image

    from app.storage import image_dir, save_image_files, thumb_dir

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "PNG", exif=b"esto no es una cabecera TIFF")
    v = validate_image_bytes(buf.getvalue())  # pasa la validación…
    stored = save_image_files(991, v)  # …y ahora también el guardado
    assert (image_dir(991) / stored).exists() and (thumb_dir(991) / (stored.rsplit(".", 1)[0] + ".jpg")).exists()


def test_png_with_hidden_data_after_iend_is_rejected():
    with pytest.raises(InvalidImage):
        validate_image_bytes(png_bytes("blue", 3) + b"datos escondidos sin firma conocida")


def test_jpeg_with_appended_zip_is_rejected():
    with pytest.raises(InvalidImage):
        validate_image_bytes(png_bytes("blue", 4, fmt="JPEG") + ZIP)


def test_payload_inside_metadata_is_rejected():
    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", SCRIPT.decode())
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "PNG", pnginfo=info)
    with pytest.raises(InvalidImage):
        validate_image_bytes(buf.getvalue())


@pytest.mark.parametrize("data", [
    b"#EXT" + b"M3U\n#EXTINF:1,\nfile:///tmp/x\n",
    b"ffconcat " + b"version 1.0\nfile '/tmp/x'\n",
    SHELL,
])
def test_text_files_disguised_as_audio_are_rejected(data):
    with pytest.raises(InvalidImage):
        validate_audio_bytes(data, "sonido.wav")


def test_audio_with_executable_is_rejected():
    with pytest.raises(InvalidImage):
        validate_audio_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64 + PE_STUB, "sonido.wav")


def test_upload_endpoint_reports_rejection_and_stores_nothing():
    c = logged_in("ana")
    p = create_project(c)
    cat = p["categories"][0]["id"]
    r = upload(c, p["id"], cat, [
        ("limpia.png", png_bytes("red", 5), "image/png"),
        ("troyano.png", png_bytes("red", 6) + PE_STUB, "image/png"),
        ("pagina.php.png", png_bytes("red", 7) + PHP, "image/png"),
    ])
    body = r.json()
    assert r.status_code == 201
    assert len(body["uploaded"]) == 1 and len(body["rejected"]) == 2
    assert all("ejecutable" in x["reason"] for x in body["rejected"])


def test_served_files_are_sandboxed():
    c = logged_in("ana")
    p = create_project(c)
    img = upload(c, p["id"], p["categories"][0]["id"], [("a.png", png_bytes("red", 8), "image/png")]).json()["uploaded"][0]
    for path in (f"/api/images/{img['id']}/file", f"/api/images/{img['id']}/thumb"):
        h = c.get(path).headers
        assert h["x-content-type-options"] == "nosniff"
        assert "sandbox" in h["content-security-policy"]
