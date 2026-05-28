from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form
)

import os

from app.services.pdf_service import (
    extract_text_from_pdf
)

from app.services.embedding_service import (
    chunk_text,
    create_embeddings
)

from app.services.chroma_service import (
    get_or_create_collection
)

router = APIRouter()

UPLOAD_DIR = "uploads"

ALLOWED_EXTENSIONS = [".pdf"]


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Form(...)
):

    try:

        # Validate file type
        file_extension = os.path.splitext(
            file.filename
        )[1].lower()

        if file_extension not in ALLOWED_EXTENSIONS:

            return {
                "error": "Only PDF files are allowed"
            }

        # Create chat folder
        chat_folder = os.path.join(
            UPLOAD_DIR,
            chat_id
        )

        os.makedirs(
            chat_folder,
            exist_ok=True
        )

        # Save file
        file_path = os.path.join(
            chat_folder,
            file.filename
        )

        with open(file_path, "wb") as f:

            content = await file.read()

            f.write(content)

        # Extract text
        extracted_text = extract_text_from_pdf(
            file_path
        )

        # Empty PDF check
        if not extracted_text.strip():

            return {
                "error": "No text found in PDF"
            }

        # Chunk text
        chunks = chunk_text(
            extracted_text
        )

        # Create embeddings
        embeddings = create_embeddings(
            chunks
        )

        # Create collection
        collection = get_or_create_collection(
            chat_id
        )

        # Store in ChromaDB
        collection.add(
            documents=chunks,

            embeddings=embeddings,

            ids=[
                f"{chat_id}_{i}"
                for i in range(len(chunks))
            ],

            metadatas=[
                {
                    "filename": file.filename,
                    "chat_id": chat_id
                }
                for _ in range(len(chunks))
            ]
        )

        return {
            "message": "File uploaded successfully",
            "chat_id": chat_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "sample_chunk": (
                chunks[0]
                if chunks
                else "No chunks created"
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }

