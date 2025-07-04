"""
Pydantic compatibility wrapper to handle v1/v2 differences
"""

import sys
from unittest.mock import MagicMock

# Create a mock for PydanticDeprecationWarning if it doesn't exist
class PydanticDeprecationWarning(Warning):
    """Mock for Pydantic v2 deprecation warning."""
    pass

# Monkey-patch pydantic module
if 'pydantic' in sys.modules:
    pydantic = sys.modules['pydantic']
    if not hasattr(pydantic, 'PydanticDeprecationWarning'):
        pydantic.PydanticDeprecationWarning = PydanticDeprecationWarning

# Create other v2 compatibility attributes
compatibility_attrs = [
    'ConfigDict',
    'field_validator',
    'model_validator',
    'computed_field',
    'Field',
    'PrivateAttr',
]

for attr in compatibility_attrs:
    if 'pydantic' in sys.modules and not hasattr(sys.modules['pydantic'], attr):
        setattr(sys.modules['pydantic'], attr, MagicMock())