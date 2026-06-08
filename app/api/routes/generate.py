"""On-demand media + document generation:
- POST /generate/image  -> {"image": "data:image/png;base64,..."}
- POST /generate/video  -> {"url": "https://gen.pollinations.ai/video/..."}
- POST /generate/pdf    -> binary PDF download (Content-Type: application/pdf)
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user
from app.models import User
from app.services.generate_service import (
    generate_document_excel,
    generate_document_pdf,
    generate_document_ppt,
    generate_document_word,
    generate_image_b64,
    generate_video_bytes,
)

router = APIRouter(tags=["generate"])


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1500)
    width: int = 1024
    height: int = 1024


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1500)
    duration: int = 5
    model: str = "wan-fast"


class PdfRequest(BaseModel):
    # Generous cap: the prompt may be a topic OR existing content the user wants
    # converted into a document ("make this into a PDF"), which can be long.
    prompt: str = Field(..., min_length=1, max_length=8000)


@router.post("/generate/image")
async def gen_image(req: ImageRequest, user: User = Depends(get_current_user)):
    try:
        data_uri = await run_in_threadpool(
            generate_image_b64, req.prompt, req.width, req.height
        )
        return {"image": data_uri}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Image generation failed.")


@router.post("/generate/video")
async def gen_video(req: VideoRequest, user: User = Depends(get_current_user)):
    # Backend proxies the Pollinations MP4 so the secret key never reaches
    # the browser — we send Bearer server-side and stream the bytes back.
    try:
        video_bytes, content_type = await run_in_threadpool(
            generate_video_bytes, req.prompt, req.model, req.duration
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Video generation failed.")
    return Response(content=video_bytes, media_type=content_type)


@router.post("/generate/pdf")
async def gen_pdf(req: PdfRequest, user: User = Depends(get_current_user)):
    try:
        title, pdf_bytes = await run_in_threadpool(generate_document_pdf, req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="PDF generation failed.")

    # Make the filename safe for the Content-Disposition header.
    safe = "".join(c for c in (title or "document") if c.isalnum() or c in "-_ ").strip() or "document"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe[:80]}.pdf"',
        },
    )


class DocRequest(BaseModel):
    # Generous cap: prompt may be a topic OR existing content to convert into a
    # spreadsheet / document / deck ("turn this table into Excel"), often long.
    prompt: str = Field(..., min_length=1, max_length=8000)


@router.post("/generate/excel")
async def gen_excel(req: DocRequest, user: User = Depends(get_current_user)):
    try:
        title, xlsx_bytes = await run_in_threadpool(generate_document_excel, req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Excel generation failed.")
    safe = "".join(c for c in (title or "spreadsheet") if c.isalnum() or c in "-_ ").strip() or "spreadsheet"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe[:80]}.xlsx"'},
    )


@router.post("/generate/word")
async def gen_word(req: DocRequest, user: User = Depends(get_current_user)):
    try:
        title, docx_bytes = await run_in_threadpool(generate_document_word, req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Word document generation failed.")
    safe = "".join(c for c in (title or "document") if c.isalnum() or c in "-_ ").strip() or "document"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe[:80]}.docx"'},
    )


@router.post("/generate/ppt")
async def gen_ppt(req: DocRequest, user: User = Depends(get_current_user)):
    try:
        title, pptx_bytes = await run_in_threadpool(generate_document_ppt, req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="PowerPoint generation failed.")
    safe = "".join(c for c in (title or "presentation") if c.isalnum() or c in "-_ ").strip() or "presentation"
    return Response(
        content=pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{safe[:80]}.pptx"'},
    )
