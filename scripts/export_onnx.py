import logging

from flow3r.models.flow3r import Flow3r

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flow3r = Flow3r.export_onnx()
