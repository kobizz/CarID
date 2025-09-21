"""Test cases for Pydantic models"""
import pytest
from service.models import ClassifyReq, ClassifyResp, AddReq


def test_classify_request_basic():
    """Test basic ClassifyReq creation"""
    req = ClassifyReq()
    assert req.image_b64 is None
    assert req.image_url is None
    assert req.topk is None
    assert req.accept_threshold is None
    assert req.margin_threshold is None
    assert req.crop_norm is None
    assert req.debug is False


def test_classify_request_with_data():
    """Test ClassifyReq with actual data"""
    req = ClassifyReq(
        image_b64="data:image/jpeg;base64,test",
        topk=5,
        accept_threshold=0.8,
        debug=True
    )
    assert req.image_b64 == "data:image/jpeg;base64,test"
    assert req.topk == 5
    assert req.accept_threshold == 0.8
    assert req.debug is True


def test_classify_response():
    """Test ClassifyResp creation"""
    resp = ClassifyResp(
        label="test_category",
        make="test",
        model="category", 
        score=0.95,
        accepted=True,
        topk=[("test_category", 0.95)],
        debug=None
    )
    assert resp.label == "test_category"
    assert resp.make == "test"
    assert resp.model == "category"
    assert resp.score == 0.95
    assert resp.accepted is True
    assert len(resp.topk) == 1


def test_add_request():
    """Test AddReq creation"""
    req = AddReq(
        label="test_category",
        is_negative=False,
        image_b64="data:image/jpeg;base64,test"
    )
    assert req.label == "test_category"
    assert req.is_negative is False
    assert req.image_b64 == "data:image/jpeg;base64,test"
    assert req.image_url is None
