"""Request and response models of the service API.

The backend is the system of record: with every request it sends a snapshot of the reconciled trial
records (`Dataset`). The agent only reads that snapshot, so the service itself is stateless.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LogStatus = Literal["info", "processing", "success", "error"]

TrialStatus = Literal["consistent", "single_source", "incomplete", "conflict", "manual"]


# --- Dataset snapshot -------------------------------------------------------------------------


class ReferenceEntry(BaseModel):
    """A canonical vocabulary value (crop, product, country, trial type) and the spellings that map to it."""

    model_config = ConfigDict(extra="ignore")

    name: str
    aliases: list[str] = []
    code: str | None = None


class CustomFieldRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    label: str
    type: str = "text"
    unit: str | None = None
    options: list[str] = []


class ReferenceData(BaseModel):
    crops: list[ReferenceEntry] = []
    products: list[ReferenceEntry] = []
    countries: list[ReferenceEntry] = []
    trial_types: list[ReferenceEntry] = []
    custom_fields: list[CustomFieldRef] = []


class SourceRef(BaseModel):
    """One source document or row that reported the trial, with the values it reported."""

    model_config = ConfigDict(extra="allow")

    file: str
    kind: str = "csv"
    locator: str | None = None
    values: dict[str, str] = {}


class ConflictCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: Any = None
    display: str = ""
    sources: list[str] = []


class TrialConflict(BaseModel):
    """A field on which the sources disagree."""

    model_config = ConfigDict(extra="allow")

    field: str
    label: str = ""
    status: Literal["open", "resolved"] = "open"
    candidates: list[ConflictCandidate] = []
    chosen_display: str | None = None


class TrialNote(BaseModel):
    source: str
    text: str


class DatasetTrial(BaseModel):
    """A reconciled trial. Yields are in the dataset's display unit; derived values are precomputed."""

    model_config = ConfigDict(extra="allow")

    id: str
    crop: str | None = None
    product: str | None = None
    country: str | None = None
    year: int | None = None
    trial_type: str | None = None
    treatment_yield: float | None = None
    control_yield: float | None = None
    yield_unit: str = "t/ha"
    yield_difference: float | None = None
    uplift_pct: float | None = None
    # Lowest and highest uplift the conflicting source values allow, when yields conflict.
    uplift_range: list[float] | None = None
    status: str = "consistent"
    caveats: list[str] = []
    conflicts: list[TrialConflict] = []
    missing_fields: list[str] = []
    sources: list[SourceRef] = []
    notes: list[TrialNote] = []
    custom_fields: dict[str, Any] = {}


class Dataset(BaseModel):
    display_unit: str = "t/ha"
    trials: list[DatasetTrial] = []
    reference: ReferenceData = ReferenceData()


# --- Agent ------------------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentOptions(BaseModel):
    """Per-request overrides of the service defaults. `None` means "use the service default"."""

    model: str | None = None
    ollama_base_url: str | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    num_ctx: int | None = Field(None, ge=2048, le=262144)
    think: bool | None = None
    max_tool_iterations: int | None = Field(None, ge=1, le=20)
    max_revisions: int | None = Field(None, ge=0, le=5)
    enable_reflection: bool | None = None
    offline_fallback: bool | None = None
    # Skip Ollama entirely and use the rule-based planner (useful for offline presentations).
    force_offline: bool = False
    history_turns: int = Field(6, ge=0, le=40)
    extra_instructions: str = Field("", max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []
    dataset: Dataset
    options: AgentOptions = AgentOptions()


class AgentEvent(BaseModel):
    """One step of the agent run, shown in the UI's "Inspection" tab."""

    type: Literal["log", "answer", "error", "done"]
    status: LogStatus = "info"
    message: str = ""
    node: str | None = None
    elapsed_ms: int = 0
    data: dict[str, Any] | None = None


class Verification(BaseModel):
    passed: bool
    skipped: bool = False
    issues: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["llm", "offline"]
    model: str
    fallback_reason: str | None = None
    trial_ids: list[str] = []
    sources: list[str] = []
    iterations: int
    revisions: int
    tool_calls: int
    verification: Verification
    trace: list[AgentEvent] = []
    duration_ms: int


# --- Report extraction -----------------------------------------------------------------------


class ReportExtraction(BaseModel):
    """Structured fields the LLM extracts from a free-text trial report."""

    trial_id: str | None = Field(None, description="Trial identifier exactly as written, e.g. 'T07'.")
    crop: str | None = Field(None, description="Crop name, e.g. 'Wheat'.")
    product: str | None = Field(None, description="Product tested in the trial, e.g. 'Harvest Plus'.")
    country: str | None = Field(None, description="Country where the trial ran.")
    year: int | None = Field(None, description="Trial year as a four-digit number.")
    trial_type: str | None = Field(None, description="Trial type, e.g. 'Scientific' or 'Demonstration'.")
    treatment_yield: float | None = Field(None, description="Yield of the treated plots, number only.")
    control_yield: float | None = Field(None, description="Yield of the untreated control, number only.")
    yield_unit: str | None = Field(None, description="Unit of both yields, e.g. 't/ha' or 'kg/ha'.")
    notes: str | None = Field(None, description="Caveats about replication, sites or statistics, quoted from the text.")


class ExtractRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=50000)
    reference: ReferenceData = ReferenceData()
    units: list[str] = ["t/ha", "kg/ha"]
    options: AgentOptions = AgentOptions()


class ExtractResponse(BaseModel):
    ok: bool
    mode: Literal["llm", "unavailable"]
    model: str
    fields: ReportExtraction | None = None
    issues: list[str] = []
    attempts: int = 0
    trace: list[AgentEvent] = []


class LlmTestRequest(BaseModel):
    model: str | None = None
    ollama_base_url: str | None = None


class OllamaStatusOut(BaseModel):
    base_url: str
    model: str
    reachable: bool
    model_available: bool
    available_models: list[str] = []
    error: str | None = None
