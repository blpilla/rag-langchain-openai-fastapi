import pytest
from src.document_processor import DocumentProcessor

def test_document_segmentation():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    document = "Este é um documento de teste. " * 10  # 290 caracteres
    segments = processor.process(document)
    
    assert len(segments) > 1, "O documento deveria ser dividido em múltiplos segmentos"
    assert all(len(seg) <= 100 for seg in segments), "Todos os segmentos devem ter no máximo 100 caracteres"
    
    # Verifica a sobreposição
    for i in range(len(segments) - 1):
        overlap = set(segments[i].split()) & set(segments[i+1].split())
        assert len(overlap) > 0, "Deve haver alguma sobreposição entre segmentos adjacentes"

def test_empty_document():
    processor = DocumentProcessor()
    segments = processor.process("")
    assert len(segments) == 0, "Um documento vazio deve resultar em uma lista vazia de segmentos"

def test_short_document():
    processor = DocumentProcessor(chunk_size=1000)
    short_doc = "Este é um documento curto."
    segments = processor.process(short_doc)
    assert len(segments) == 1, "Um documento curto não deve ser segmentado"
    assert segments[0] == short_doc, "O segmento único deve ser igual ao documento original"