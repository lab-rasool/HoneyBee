from .Slide.slide import Slide


def __getattr__(name):
    if name == "manifest_to_df":
        from .Reader.mindsDBreader import manifest_to_df

        return manifest_to_df
    if name == "PDF":
        from .Reader.reader import PDF

        return PDF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
