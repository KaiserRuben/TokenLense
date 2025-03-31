from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class DataVersion(str, Enum):
    V1_0_0 = "1.0.0"
    V1_0_1 = "1.0.1"
    # ... if needed


class TokenData(BaseModel):
    """Raw token information"""
    token: str
    token_id: int
    clean_token: str = Field(..., description="Token with special characters removed")

    model_config = ConfigDict(
        protected_namespaces=()
    )

    @property
    def cleaned(self) -> str:
        """Get clean token value"""
        return self.clean_token or self.token.lstrip('Ġ').strip()


class AssociationData(BaseModel):
    """Token association analysis results"""
    input_tokens: List[TokenData]
    output_tokens: List[TokenData]
    association_matrix: List[List[float]]
    normalized_association: Optional[List[List[float]]] = None

    model_config = ConfigDict(
        protected_namespaces=()
    )


class AnalysisMetadata(BaseModel):
    """Metadata for analysis results"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    llm_id: str = Field(..., description="Model identifier")
    llm_version: str = Field(..., description="Model version")
    prompt: str
    generation_params: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    version: DataVersion = Field(default=DataVersion.V1_0_0)

    model_config = ConfigDict(
        protected_namespaces=()
    )


class AnalysisResult(BaseModel):
    """Complete analysis result including data and metadata"""
    metadata: AnalysisMetadata
    data: AssociationData

    model_config = ConfigDict(
        protected_namespaces=(),
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Schema migration utilities
class SchemaMigration:
    """Handles migration between schema versions"""

    @classmethod
    def migrate(cls, data: Dict, target_version: DataVersion) -> Dict:
        """Migrate data to target version"""
        current_version = DataVersion(data["metadata"]["version"])

        if current_version == target_version:
            return data

        raise ValueError(
            f"Unsupported migration path: {current_version} -> {target_version}"
        )