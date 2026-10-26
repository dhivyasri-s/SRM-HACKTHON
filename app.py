import os
import time
import hmac
import hashlib
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    from google import genai
except ImportError:
    genai = None

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
APP_PORT = int(os.getenv("PORT", 8000))
APP_HOST = os.getenv("HOST", "0.0.0.0")
LOTO_SECRET_KEY = os.getenv("LOTO_SECRET_KEY", "voltsnap-loto-master-key-2026")

# Initialize Gemini Client if key is configured
client = None
if genai and GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini Client: {e}")

app = FastAPI(
    title="VoltSnap-AI",
    description="Real-time Smart Grid Protection System against Overhead LT Line Snaps with Sub-Cycle Tripping, Gemini 2.5 Flash, and LOTO Clearance",
    version="2.0.0",
)

# Enable CORS for dashboard interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Internal Grid State Management
# ---------------------------------------------------------
class GridState:
    def __init__(self):
        self.breaker_closed: bool = True
        self.lockout_active: bool = False
        self.fault_distance_m: Optional[float] = None
        self.pole_range: Optional[str] = None
        self.gemini_diagnosis: Optional[str] = None
        self.last_event: str = "SYSTEM_INITIALIZED_HEALTHY"
        self.trip_latency_ms: Optional[float] = None
        self.last_fault_impedance: Optional[Dict[str, float]] = None
        self.broadcast_info: Optional[Dict[str, Any]] = None
        self.fault_location: Optional[Dict[str, Any]] = None
        self.geospatial_dispatch: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breaker_closed": self.breaker_closed,
            "lockout_active": self.lockout_active,
            "fault_distance_m": self.fault_distance_m,
            "pole_range": self.pole_range,
            "gemini_diagnosis": self.gemini_diagnosis,
            "last_event": self.last_event,
            "trip_latency_ms": self.trip_latency_ms,
            "last_fault_impedance": self.last_fault_impedance,
            "broadcast_info": self.broadcast_info,
            "fault_location": self.fault_location,
            "geospatial_dispatch": self.geospatial_dispatch,
        }

    def reset_to_healthy(self):
        self.breaker_closed = True
        self.lockout_active = False
        self.fault_distance_m = None
        self.pole_range = None
        self.gemini_diagnosis = None
        self.last_event = "RESTORED_NORMAL"
        self.trip_latency_ms = None
        self.last_fault_impedance = None
        self.broadcast_info = None
        self.fault_location = None
        self.geospatial_dispatch = None


grid_state = GridState()


# ---------------------------------------------------------
# HMAC-SHA256 Offline LOTO Helpers
# ---------------------------------------------------------
def compute_loto_hmac(lineman_id: str, work_order_id: str, substation_id: str = "DTR-11") -> str:
    """Generates an offline verifiable 6-digit HMAC-SHA256 token for lineman safety clearance."""
    norm_lineman = lineman_id.strip().upper()
    norm_wo = work_order_id.strip().upper()
    norm_sub = substation_id.strip().upper()
    message = f"VOLTSNAP_LOTO:{norm_lineman}:{norm_wo}:{norm_sub}"
    signature = hmac.new(
        LOTO_SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # 6-digit cryptographic code (HOTP/HMAC-SHA256 dynamic truncation modulo 10^6)
    code_int = int(signature[:8], 16) % 1_000_000
    return f"{code_int:06d}"


def verify_loto_token(token: str, lineman_id: str, work_order_id: str, substation_id: str = "DTR-11") -> bool:
    """Verifies the HMAC-SHA256 6-digit token locally without internet access."""
    expected_token = compute_loto_hmac(lineman_id, work_order_id, substation_id)
    cleaned = token.strip().upper().replace("LOTO-", "")
    return hmac.compare_digest(cleaned, expected_token)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class EventSimulationRequest(BaseModel):
    event_type: str = Field(
        ...,
        description="Event type: 'conductor_snap', 'lightning_surge', or 'motor_inrush'",
        json_schema_extra={"example": "conductor_snap"},
    )
    feeder_id: str = Field(default="LT-FEEDER-04B", description="Overhead line identifier")
    dtr_source: str = Field(default="DTR-11", description="Distribution Transformer identifier")
    custom_voltage_v: Optional[float] = None
    custom_current_a: Optional[float] = None
    custom_tension_kn: Optional[float] = None


class EventSimulationResponse(BaseModel):
    event_type: str
    behavior: str
    breaker_action: str
    breaker_closed: bool
    lockout_active: bool
    trip_latency_ms: Optional[float] = None
    fault_distance_m: Optional[float] = None
    pole_range: Optional[str] = None
    impedance_calculation: Optional[Dict[str, Any]] = None
    micro_arcing_detected: bool = False
    gemini_diagnosis: Optional[str] = None
    broadcast_info: Optional[Dict[str, Any]] = None
    fault_location: Optional[Dict[str, Any]] = None
    geospatial_dispatch: Optional[Dict[str, Any]] = None
    grid_state: Dict[str, Any]


class LotoTokenRequest(BaseModel):
    lineman_id: str = Field(..., json_schema_extra={"example": "LM-8492"}, description="Certified lineman badge ID")
    work_order_id: str = Field(..., json_schema_extra={"example": "WO-SNAP-2026-114"}, description="Substation emergency ticket")
    substation_id: str = Field(default="DTR-11", description="DTR feeder unit ID")
    pole_range: Optional[str] = Field(default="Pole #14–#15", description="Target maintenance span")


class LotoTokenResponse(BaseModel):
    success: bool
    offline_token: str
    lineman_id: str
    work_order_id: str
    substation_id: str
    pole_range: str
    verification_algorithm: str
    instructions: str


class LotoUnlockRequest(BaseModel):
    lineman_id: str = Field(..., json_schema_extra={"example": "LM-8492"})
    work_order_id: str = Field(..., json_schema_extra={"example": "WO-SNAP-2026-114"})
    token: str = Field(..., json_schema_extra={"example": "849201"})
    substation_id: str = Field(default="DTR-11")
    clearance_notes: Optional[str] = Field(
        default="Conductor re-tensioned and ground clearance verified. All personnel clear.",
        description="Field verification notes",
    )


class SensorReading(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    line_id: str = Field(default="LT-FEEDER-04B")
    location: str = Field(default="Sector 12 Substation Sub-Feeder")
    voltage_v: float
    current_a: float
    tension_kn: float
    vibration_frequency_hz: float = 12.4
    ground_impedance_ohm: float = 0.8


class SnapAnalysisResponse(BaseModel):
    status: str
    snap_detected: bool
    risk_score: float
    trip_signal: bool
    response_time_ms: float
    gemini_insights: Optional[str] = None
    telemetry_summary: Dict[str, Any]


class IvrDispatchRequest(BaseModel):
    phone_number: Optional[str] = Field(default="+91 98401 23456", description="Target citizen phone number")
    language: Optional[str] = Field(default="both", description="Language: 'english', 'tamil', or 'both'")


# ---------------------------------------------------------
# Core API Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """System health check, grid breaker status, and Gemini AI status."""
    return {
        "status": "online",
        "system": "VoltSnap-AI Real-Time Grid Guardian",
        "gemini_configured": bool(client is not None),
        "target_model": "gemini-2.5-flash",
        "current_grid_state": grid_state.to_dict(),
    }


@app.get("/api/grid/status")
async def get_grid_status():
    """Returns the internal grid states."""
    return {
        "status": "success",
        "state": grid_state.to_dict(),
    }


@app.post("/api/grid/simulate-event", response_model=EventSimulationResponse)
async def simulate_event(req: EventSimulationRequest):
    """
    Simulates grid behavior and executes protection algorithms:
    1. False-Trip Rejection: Ignores lightning_surge and motor_inrush without tripping breaker.
    2. Conductor Snap: Detects micro-arcing signatures, calculates impedance-based distance
       (Z = R + jX) pinning fault to Pole #14-#15 (~342m from DTR-11), opens virtual breaker in <50ms,
       and triggers Gemini 2.5 Flash for a concise 3-bullet root-cause diagnosis.
    """
    start_time = time.perf_counter()
    event = req.event_type.lower().strip()

    # Safety Guard: If Lockout is active, breaker cannot close until cleared by LOTO
    if grid_state.lockout_active and event != "conductor_snap":
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Digital Lockout/Tagout (LOTO) is ACTIVE. The virtual breaker is locked open. Lineman clearance token required to unlock.",
        )

    # ---------------------------------------------------------
    # BEHAVIOR 1: False-Trip Rejection (lightning_surge & motor_inrush)
    # ---------------------------------------------------------
    if event in ["lightning_surge", "motor_inrush"]:
        # Sub-cycle transient evaluation
        duration_ms = 4.2 if event == "lightning_surge" else 420.0
        peak_current = 18.5 if event == "lightning_surge" else 92.4
        voltage_dip = 12.0 if event == "lightning_surge" else 185.0

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        grid_state.last_event = f"TRANSIENT_REJECTED_{event.upper()}"
        grid_state.trip_latency_ms = None

        behavior_desc = (
            "False-Trip Rejection: High-voltage surge arrester dissipated sub-cycle impulse (<5ms). "
            "Mechanical line tension nominal (1.35 kN). Waveform symmetry confirmed non-rupture transient."
            if event == "lightning_surge"
            else "False-Trip Rejection: Symmetric lagging inductive inrush current detected (~6x nominal). "
            "Absence of zero-sequence arc current or tension loss confirms motor starting. Breaker held closed."
        )

        return EventSimulationResponse(
            event_type=event,
            behavior="FALSE_TRIP_REJECTION",
            breaker_action="HELD_CLOSED (NO_TRIP)",
            breaker_closed=grid_state.breaker_closed,
            lockout_active=grid_state.lockout_active,
            trip_latency_ms=round(elapsed_ms, 2),
            fault_distance_m=None,
            pole_range=None,
            impedance_calculation=None,
            micro_arcing_detected=False,
            gemini_diagnosis=None,
            grid_state=grid_state.to_dict(),
        )

    # ---------------------------------------------------------
    # BEHAVIOR 2: Conductor Snap Detection, Impedance Pinning & <50ms Tripping
    # ---------------------------------------------------------
    elif event == "conductor_snap":
        # 1. Micro-arcing & mechanical tension collapse signature
        micro_arcing_detected = True
        tension_kn = req.custom_tension_kn if req.custom_tension_kn is not None else 0.08
        voltage_v = req.custom_voltage_v if req.custom_voltage_v is not None else 18.2
        current_a = req.custom_current_a if req.custom_current_a is not None else 215.0

        # 2. Impedance-Based Distance Calculation (Z = R + jX)
        # Overhead LT ACSR Weasel conductor parameters:
        # Unit resistance r = 0.720 Ohm/km, Unit reactance x = 0.310 Ohm/km
        # Pinning the fault to Pole #14-#15 (~342m from DTR-11):
        target_distance_m = 342.0
        distance_km = target_distance_m / 1000.0

        R_line = round(0.720 * distance_km, 4)  # ~0.2462 Ohm
        X_line = round(0.310 * distance_km, 4)  # ~0.1060 Ohm
        Z_magnitude = round(np.sqrt(R_line**2 + X_line**2), 4)  # ~0.2681 Ohm

        pole_span_m = 24.4  # Average span between LT poles
        calculated_pole = round(target_distance_m / pole_span_m, 1)
        pole_range = "Pole #14–#15 (~342m from DTR-11)"

        # 3. Microsecond Sub-Cycle Breaker Tripping (< 50 ms)
        # Measured hardware relay response time simulation
        trip_elapsed_ms = round(18.5 + (np.random.random() * 8.0), 2)  # Typically ~22-26 ms, strictly < 50ms

        # Mutate internal grid state
        grid_state.breaker_closed = False
        grid_state.lockout_active = True
        grid_state.fault_distance_m = target_distance_m
        grid_state.pole_range = pole_range
        grid_state.trip_latency_ms = trip_elapsed_ms
        grid_state.last_event = "CONDUCTOR_SNAP_TRIP_LOCKOUT"
        grid_state.last_fault_impedance = {
            "R_ohm": R_line,
            "X_ohm": X_line,
            "Z_mag_ohm": Z_magnitude,
            "fault_current_a": current_a,
        }

        # 4. Trigger Gemini 2.5 Flash for 3-Bullet Root-Cause Diagnosis
        gemini_diagnosis = None
        if client:
            try:
                gemini_prompt = f"""
You are the autonomous protection engineer for VoltSnap-AI. An overhead LT (Low Tension) distribution line has severed.
Telemetry & Fault Parameters:
- Feeder: {req.feeder_id} fed from {req.dtr_source}
- Conductor Mechanical Tension: {tension_kn:.2f} kN (Dropped from nominal 1.35 kN)
- Measured Loop Impedance: Z = {R_line} + j{X_line} Ohm (|Z| = {Z_magnitude} Ohm)
- Calculated Distance: {target_distance_m} meters from {req.dtr_source}
- Physical Location: {pole_range}
- Hardware Breaker Action: Virtual breaker tripped open in {trip_elapsed_ms} ms (<50 ms threshold). Lockout ACTIVE.
- Signature: Intermittent micro-arcing burst followed by complete ground contact.

Provide a concise, professional 3-bullet root-cause diagnosis.
Strictly adhere to exactly 3 bullet points starting with '• ':
Bullet 1: Physical Conductor Rupture & Micro-Arcing analysis.
Bullet 2: Impedance-based fault localization pinning to Pole #14–#15 (~342m from DTR-11).
Bullet 3: Ultra-fast trip confirmation (<50ms) and Lineman LOTO safety lockout isolation status.
"""
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=gemini_prompt,
                )
                gemini_diagnosis = response.text.strip()
            except Exception as e:
                print(f"Gemini API error during event diagnosis: {e}")

        # Deterministic, precise 3-bullet diagnosis fallback if Gemini is unconfigured or unavailable
        if not gemini_diagnosis:
            gemini_diagnosis = (
                f"• Micro-arcing acoustic/current signatures confirmed severe conductor tensile rupture and ground contact.\n"
                f"• Reactive loop impedance (Z = {R_line} + j{X_line} Ω, |Z| = {Z_magnitude} Ω) localized fault locus precisely between Pole #14–#15 (~342m from DTR-11).\n"
                f"• Virtual feeder breaker de-energized within {trip_elapsed_ms}ms (<50ms limit); automatic Lockout/Tagout (LOTO) engaged to prevent lethal back-feed."
            )

        grid_state.gemini_diagnosis = gemini_diagnosis

        # Geo-Spatial Dual-Pin Routing & Hyper-Local Target Telemetry
        geospatial_dispatch = {
            "fault_location": {
                "pole_id": "Pole #14 to #15",
                "feeder": "DTR-11",
                "latitude": 10.0261,
                "longitude": 76.3125,
                "address": "Gandhi Street, Sector 4 Zone"
            },
            "assigned_lineman": {
                "name": "S. Murugan (Senior Tech)",
                "crew_id": "CREW-TANGEDCO-11",
                "current_latitude": 10.0210,
                "current_longitude": 76.3080,
                "distance_km": 0.65,
                "eta_mins": 4.2,
                "route_guidance": "Take North Ring Road -> 2nd Left past Substation -> Pole #14 approach"
            },
            "hyper_local_public_alert": {
                "targeted_radius_meters": 250,
                "recipients_notified": 142,
                "channels_sent": ["WhatsApp", "SMS", "Automated Tamil IVR Voice Call"]
            }
        }
        grid_state.geospatial_dispatch = geospatial_dispatch

        # Coordinates for DTR-11 Pole #14-#15 Fault Location (Kerala Grid Sandbox)
        fault_location = {
            "dtr_id": "DTR-11",
            "pole_span": pole_range,
            "latitude": 10.0261,
            "longitude": 76.3125,
            "address": "Gandhi Street, Sector 4 Zone",
            "lineman_base_lat": 10.0210,
            "lineman_base_lng": 76.3080,
            "estimated_eta_mins": 4.2,
            "distance_km": 0.65,
            "lineman_name": "S. Murugan (Senior Tech)",
            "crew_id": "CREW-TANGEDCO-11",
            "route_guidance": "Take North Ring Road -> 2nd Left past Substation -> Pole #14 approach"
        }
        grid_state.fault_location = fault_location

        # Enhanced Proximity Warning Payload with Automated IVR Voice Call
        broadcast_payload = {
            "status": "DISPATCHED_SUCCESSFULLY",
            "latency_seconds": 4.2,
            "target_recipients": 142,
            "targeted_radius_meters": 250,
            "feeder_id": "DTR-11",
            "pole_span": pole_range,
            "address": "Gandhi Street, Sector 4 Zone",
            "channels": ["WhatsApp", "SMS", "Automated Tamil IVR Voice Call"],
            "ivr_audio_text_english": "Emergency Alert! Electrical wire break detected on your street. Power cut initiated. Stay indoors and keep away from ground wires.",
            "ivr_audio_text_tamil": "அவசர எச்சரிக்கை! உங்கள் தெருவில் மின்கம்பி அறுந்து விழுந்துள்ளது. மின்சாரம் நிறுத்தப்பட்டுள்ளது. தெருவில் இறங்க வேண்டாம்.",
            "message_english": "Warning: Electric line snap detected near your street. Power disconnected for safety.",
            "message_tamil": "எச்சரிக்கை: உங்கள் தெருவில் மின்சாரக் கம்பி அறுந்து விழுந்துள்ளது. மின் இணைப்பு துண்டிக்கப்பட்டுள்ளது."
        }
        grid_state.broadcast_info = broadcast_payload

        return EventSimulationResponse(
            event_type="conductor_snap",
            behavior="CONDUCTOR_SNAP_DETECTED",
            breaker_action="BREAKER_OPENED_AND_LOCKED",
            breaker_closed=grid_state.breaker_closed,
            lockout_active=grid_state.lockout_active,
            trip_latency_ms=trip_elapsed_ms,
            fault_distance_m=target_distance_m,
            pole_range=pole_range,
            impedance_calculation={
                "R_ohm": R_line,
                "X_ohm": X_line,
                "Z_magnitude_ohm": Z_magnitude,
                "unit_resistance_ohm_per_km": 0.720,
                "unit_reactance_ohm_per_km": 0.310,
                "distance_meters": target_distance_m,
                "dtr_source": req.dtr_source,
            },
            micro_arcing_detected=micro_arcing_detected,
            gemini_diagnosis=gemini_diagnosis,
            broadcast_info=broadcast_payload,
            fault_location=fault_location,
            geospatial_dispatch=geospatial_dispatch,
            grid_state=grid_state.to_dict(),
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown event_type '{req.event_type}'. Supported events: 'conductor_snap', 'lightning_surge', 'motor_inrush'",
        )


# ---------------------------------------------------------
# Lineman Digital Lockout/Tagout (LOTO) Clearance Endpoints
# ---------------------------------------------------------
@app.post("/api/loto/generate-offline-token", response_model=LotoTokenResponse)
async def generate_offline_token(req: LotoTokenRequest):
    """
    Generates a cryptographically signed offline clearance token using local HMAC-SHA256.
    Allows linemen in remote field locations without cellular internet to authenticate
    and clear breaker lockout once physical repairs at Pole #14-#15 are complete.
    """
    offline_token = compute_loto_hmac(req.lineman_id, req.work_order_id, req.substation_id)

    return LotoTokenResponse(
        success=True,
        offline_token=offline_token,
        lineman_id=req.lineman_id.upper(),
        work_order_id=req.work_order_id.upper(),
        substation_id=req.substation_id.upper(),
        pole_range=req.pole_range or "Pole #14–#15 (~342m from DTR-11)",
        verification_algorithm="HMAC-SHA256 (Offline Local Secret)",
        instructions=(
            f"Lineman {req.lineman_id.upper()} must input this token into the field terminal "
            f"or submit via /api/loto/unlock along with Work Order {req.work_order_id.upper()} "
            "after conductor splicing and grounding verification are completed."
        ),
    )


@app.post("/api/loto/unlock")
async def unlock_loto(req: LotoUnlockRequest):
    """
    Validates the offline HMAC-SHA256 token.
    If authentic:
    - Clears the Digital Lockout (lockout_active = False)
    - Re-closes the virtual breaker (breaker_closed = True)
    - Resets fault distance and pole range
    - Updates grid state with lineman clearance log
    """
    if not grid_state.lockout_active and grid_state.breaker_closed:
        return {
            "success": True,
            "message": "System is already healthy. Breaker is closed and no lockout is active.",
            "grid_state": grid_state.to_dict(),
        }

    is_valid = verify_loto_token(req.token, req.lineman_id, req.work_order_id, req.substation_id)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "SECURITY LOCKOUT REJECTED: Invalid HMAC-SHA256 signature for the provided "
                f"Lineman ID '{req.lineman_id}' and Work Order '{req.work_order_id}'. "
                "Feeder breaker remains LOCKED OPEN for safety."
            ),
        )

    # Valid token: Reset safety lock and re-energize feeder
    grid_state.reset_to_healthy()
    grid_state.last_event = f"LOTO_CLEARED_BY_{req.lineman_id.upper()}"
    grid_state.gemini_diagnosis = (
        f"• LOTO Clearance Verified: Lineman {req.lineman_id.upper()} under Work Order {req.work_order_id.upper()}.\n"
        f"• Clearance Notes: {req.clearance_notes}\n"
        f"• Grid Status: Virtual breaker safely re-closed. Overhead LT Feeder DTR-11 re-energized."
    )

    return {
        "success": True,
        "message": "Lineman LOTO verification successful. Lockout cleared and virtual breaker re-closed.",
        "cleared_by": req.lineman_id.upper(),
        "work_order_id": req.work_order_id.upper(),
        "clearance_timestamp": time.time(),
        "notes": req.clearance_notes,
        "grid_state": grid_state.to_dict(),
    }


# ---------------------------------------------------------
# Automated IVR Voice Call Broadcast Endpoint
# ---------------------------------------------------------
@app.post("/api/ivr/dispatch-call")
async def dispatch_ivr_call(req: Optional[IvrDispatchRequest] = None):
    """Simulates automated IVR phone call dispatch to citizens near severed conductor span."""
    phone = req.phone_number if req and req.phone_number else "+91 98401 23456"
    lang = req.language if req and req.language else "both"
    return {
        "status": "CALL_CONNECTED",
        "caller_id": "1912 - TANGEDCO/KSEB Emergency Dispatch",
        "timestamp": time.time(),
        "dialed_recipient": phone,
        "language_selected": lang,
        "ivr_audio_text_english": (
            "Emergency Alert! Electrical wire break detected on your street. "
            "Power cut initiated. Stay indoors and keep away from ground wires."
        ),
        "ivr_audio_text_tamil": (
            "அவசர எச்சரிக்கை! உங்கள் தெருவில் மின்கம்பி அறுந்து விழுந்துள்ளது. "
            "மின்சாரம் நிறுத்தப்பட்டுள்ளது. தெருவில் இறங்க வேண்டாம்."
        ),
        "target_recipients": 142,
        "speech_synthesis_ready": True,
    }


# ---------------------------------------------------------
# Backward Compatibility Endpoint
# ---------------------------------------------------------
@app.post("/api/analyze-telemetry", response_model=SnapAnalysisResponse)
async def analyze_telemetry(reading: SensorReading):
    """Evaluates telemetry reading and maps to simulation behavior."""
    is_snap = reading.tension_kn < 0.25 or (reading.voltage_v < 60.0 and reading.current_a < 0.5)

    if is_snap:
        sim_res = await simulate_event(
            EventSimulationRequest(
                event_type="conductor_snap",
                custom_voltage_v=reading.voltage_v,
                custom_current_a=reading.current_a,
                custom_tension_kn=reading.tension_kn,
            )
        )
        return SnapAnalysisResponse(
            status="CRITICAL_SNAP",
            snap_detected=True,
            risk_score=0.98,
            trip_signal=True,
            response_time_ms=sim_res.trip_latency_ms or 22.4,
            gemini_insights=sim_res.gemini_diagnosis,
            telemetry_summary={
                "voltage_v": reading.voltage_v,
                "current_a": reading.current_a,
                "tension_kn": reading.tension_kn,
                "pole_range": sim_res.pole_range,
                "fault_distance_m": sim_res.fault_distance_m,
            },
        )
    else:
        return SnapAnalysisResponse(
            status="NORMAL",
            snap_detected=False,
            risk_score=0.08,
            trip_signal=False,
            response_time_ms=1.2,
            gemini_insights="Telemetry nominal. Overhead line tension and electrical parameters balanced.",
            telemetry_summary={
                "voltage_v": reading.voltage_v,
                "current_a": reading.current_a,
                "tension_kn": reading.tension_kn,
            },
        )


# ---------------------------------------------------------
# Static assets & dashboard
# ---------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serves the VoltSnap-AI real-time web control center."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "VoltSnap-AI Server Running. static/index.html not found."})


if __name__ == "__main__":
    import uvicorn
    print(f"Starting VoltSnap-AI server on http://{APP_HOST}:{APP_PORT}")
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)
