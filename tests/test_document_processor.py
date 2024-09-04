import pytest
from src.document_processor import DocumentProcessor, DocumentProcessingError

def test_process_single_document():
    # Testa o processamento de um único documento
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    content = "This is a test document. " * 10
    filename = "test_document.txt"
    
    # Simula um arquivo convertendo o conteúdo para bytes
    content_bytes = content.encode('utf-8')
    
    result = processor.process_file(content_bytes, filename)
    
    # Verifica se o documento foi dividido em múltiplos segmentos
    assert len(result) > 1, "O documento deveria ser dividido em múltiplos segmentos"
    # Verifica se cada segmento tem o conteúdo e metadados esperados
    assert all("content" in segment for segment in result)
    assert all("metadata" in segment for segment in result)
    assert all(segment["metadata"]["source"] == filename for segment in result)

def test_process_multiple_documents():
    # Testa o processamento de múltiplos documentos
    processor = DocumentProcessor()
    documents = [
        {"content": "Document 1".encode('utf-8'), "filename": "doc1.txt"},
        {"content": "Document 2".encode('utf-8'), "filename": "doc2.txt"},
        {"content": "".encode('utf-8'), "filename": "empty.txt"}
    ]
    result = processor.process_multiple_documents(documents)
    
    # Verifica se documentos vazios são ignorados
    assert len(result) == 2
    # Verifica se os metadados estão corretos para cada documento
    assert result[0]["metadata"]["source"] == "doc1.txt"
    assert result[1]["metadata"]["source"] == "doc2.txt"

def test_error_handling():
    # Testa o tratamento de erros para entradas inválidas
    processor = DocumentProcessor()
    with pytest.raises(DocumentProcessingError):
        processor.process_file(b"Invalid content", "invalid.xyz")