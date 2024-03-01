# TODO: Implement
# TODO: Test and Document
class SVSLoader:
    def __init__(self):
        pass

    def load(self, path: str):
        return path


# TODO: Implement
# TODO: Test and Document
class DICOMLoader:
    def __init__(self, embedding_model_path: str):
        self.embedding_model_path = embedding_model_path
        # Some sort of file validation here,
        # i.e. get a list of root_dir/file/paths and check if they are valid DICOM files
        # if not, log the error and process the rest
        # if all are invalid, raise an error

    def load(self, path: str):
        return path


# TODO: Implement
# TODO: Test and Document
class PDFLoader:
    def __init__(self):
        pass

    def load(self, path: str):
        return path


# TODO: Implement
# TODO: Test and Document
class MINDSLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        pass
