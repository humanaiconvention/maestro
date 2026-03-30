import os
import requests
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException, Form, Depends, Header
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

# Import from local modules
from database import SessionLocal, get_db, engine, Base
from models import (
    Actor, ActorType, HumanProfile, AgentProfile, CredentialBinding, AuthEvent, 
    ProviderType, ExperienceContribution, ProvenanceRecord, ContributionStatus,
    ActorProfile, ViabilityEvaluation, ViabilityDecision, GroundingRequest, ConsentScope, SettlementStatus
)
from exceptions import LinkingError, MaestroDomainError
from services import LinkingService, MaestroDomainService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure schema exists for local dev if Alembic isn't run
Base.metadata.create_all(bind=engine)

REQUIRED_ENVS = ["HYDRA_ADMIN_URL", "AUTHENTIK_URL"]
for env in REQUIRED_ENVS:
    if not os.getenv(env):
        logger.warning(f"Environment variable {env} is missing. Using defaults.")

HYDRA_ADMIN_URL = os.getenv("HYDRA_ADMIN_URL", "http://127.0.0.1:4445")
AUTHENTIK_URL = os.getenv("AUTHENTIK_URL", "http://127.0.0.1:9000")
SESSION_SECRET = os.getenv("SESSION_SECRET", "super-secret-session-key-dev-only")

app = FastAPI(title="Project Maestro: Identity & Domain Service")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ==========================================
# Token Validation (Hydra Introspection)
# ==========================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def verify_token(token: str = Depends(oauth2_scheme)):
    """
    Verifies that the bearer token is active and valid for the Maestro.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        resp = requests.post(
            f"{HYDRA_ADMIN_URL}/admin/oauth2/introspect",
            data={"token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        resp.raise_for_status()
        introspection = resp.json()
        
        if not introspection.get("active"):
            raise HTTPException(status_code=401, detail="Token is inactive or expired")
            
        return introspection
    except requests.RequestException as e:
        logger.error(f"Token introspection failed: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# ==========================================
# Hydra Login & Consent Nodes (Authority Layer)
# ==========================================
@app.get("/auth/login", response_class=HTMLResponse)
async def login_get(request: Request, login_challenge: str):
    """
    Hydra redirects here when an epistemic exchange requires actor identification.
    Renders the Maestro ID unified Maestro entry page.
    """
    try:
        resp = requests.get(f"{HYDRA_ADMIN_URL}/admin/oauth2/auth/requests/login", params={"login_challenge": login_challenge})
        resp.raise_for_status()
        login_request = resp.json()
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Failed to fetch login request from Hydra")

    if login_request.get("skip"):
        try:
            accept_resp = requests.put(
                f"{HYDRA_ADMIN_URL}/admin/oauth2/auth/requests/login/accept",
                params={"login_challenge": login_challenge},
                json={"subject": login_request["subject"]}
            )
            accept_resp.raise_for_status()
            return RedirectResponse(url=accept_resp.json()["redirect_to"])
        except requests.RequestException:
            raise HTTPException(status_code=500, detail="Failed to accept skipped login")

    # Generate cryptographically random CSRF token
    csrf_token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = csrf_token
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "login_challenge": login_challenge,
        "csrf_token": csrf_token
    })

@app.post("/auth/login")
async def login_post(
    request: Request,
    login_challenge: str = Form(...), 
    subject: str = Form(...), 
    csrf_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Processes the login and delegates to the LinkingService to resolve the actor.
    """
    if csrf_token != request.session.get("csrf_token"):
        return templates.TemplateResponse("login.html", {
            "request": request, "login_challenge": login_challenge, "error": "Invalid CSRF token"
        }, status_code=403)
        
    svc = LinkingService(db)
    
    try:
        # Resolve identity based on email_link fallback for this demo
        actor, is_new = svc.resolve_login_identity(
            provider_type=ProviderType.email_link,
            provider_name="email_magic_link",
            external_subject=subject,
            external_email=subject,
            current_actor_id=request.session.get("actor_id"),
            request_ip=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
        
        # Store in session
        request.session["actor_id"] = actor.actor_id

        # Accept login request in Hydra
        resp = requests.put(
            f"{HYDRA_ADMIN_URL}/admin/oauth2/auth/requests/login/accept",
            params={"login_challenge": login_challenge},
            json={"subject": actor.actor_id, "remember": True, "remember_for": 3600}
        )
        resp.raise_for_status()
        
        redirect_url = resp.json()["redirect_to"]
        
        if is_new:
            # Stash redirect URL to show onboarding first
            request.session["pending_hydra_redirect"] = redirect_url
            return RedirectResponse(url="/onboarding", status_code=302)
            
        return RedirectResponse(url=redirect_url, status_code=302)
        
    except LinkingError as e:
        return templates.TemplateResponse("login.html", {
            "request": request, "login_challenge": login_challenge, "error": str(e)
        }, status_code=400)
    except requests.RequestException as e:
        logger.error(f"Hydra login accept failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to accept login request")

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_get(request: Request):
    if "actor_id" not in request.session:
        return RedirectResponse("/auth/login")
    return templates.TemplateResponse("onboarding.html", {"request": request})

@app.post("/onboarding/skip")
async def onboarding_skip(request: Request):
    redirect_url = request.session.pop("pending_hydra_redirect", "/")
    return RedirectResponse(url=redirect_url, status_code=302)

@app.get("/auth/consent")
async def consent_get(consent_challenge: str, db: Session = Depends(get_db)):
    """
    Hydra redirects here after successful login to request permissions.
    """
    try:
        resp = requests.get(f"{HYDRA_ADMIN_URL}/admin/oauth2/auth/requests/consent", params={"consent_challenge": consent_challenge})
        resp.raise_for_status()
        consent_request = resp.json()
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Failed to fetch consent request")
    
    subject = consent_request["subject"]
    actor_profile = db.query(ActorProfile).filter(ActorProfile.actor_id == subject).first()
    
    is_eligible = actor_profile.pog_eligibility if actor_profile else False
    
    if consent_request.get("skip") or is_eligible:
        try:
            accept_resp = requests.put(
                f"{HYDRA_ADMIN_URL}/admin/oauth2/auth/requests/consent/accept",
                params={"consent_challenge": consent_challenge},
                json={
                    "grant_scope": consent_request["requested_scope"],
                    "grant_access_token_audience": consent_request["requested_access_token_audience"],
                    "session": {
                        "id_token": {"pog_eligible": is_eligible}
                    }
                }
            )
            accept_resp.raise_for_status()
            return RedirectResponse(url=accept_resp.json()["redirect_to"])
        except requests.RequestException:
            raise HTTPException(status_code=500, detail="Failed to accept consent")

    raise HTTPException(status_code=403, detail="User is not eligible for PoG consent")

# ==========================================
# Web UI: Account Settings
# ==========================================
def get_current_actor(request: Request, db: Session = Depends(get_db)):
    actor_id = request.session.get("actor_id")
    if not actor_id:
        # Fallback: check if we are in test environment or pass
        raise HTTPException(status_code=401, detail="Not logged in")
    actor = db.query(Actor).filter(Actor.actor_id == actor_id).first()
    if not actor:
        raise HTTPException(status_code=404, detail="Actor not found")
    return actor

@app.get("/account/settings", response_class=HTMLResponse)
async def settings_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    active_bindings = [b for b in actor.bindings if b.revoked_at is None]
    events = db.query(AuthEvent).filter(AuthEvent.actor_id == actor.actor_id).order_by(AuthEvent.created_at.desc()).limit(10).all()

    csrf_token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "actor": actor,
        "active_bindings": active_bindings,
        "events": events,
        "csrf_token": csrf_token
    })

@app.post("/auth/unlink")
async def unlink_provider(request: Request, binding_id: str = Form(...), csrf_token: str = Form(...), actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    svc = LinkingService(db)
    try:
        svc.unlink_provider(actor.actor_id, binding_id)
        return templates.TemplateResponse("settings.html", {
            "request": request, "actor": actor, "message": "Sign-in method removed.", 
            "active_bindings": [b for b in actor.bindings if b.revoked_at is None],
            "events": db.query(AuthEvent).filter(AuthEvent.actor_id == actor.actor_id).order_by(AuthEvent.created_at.desc()).limit(10).all()
        })
    except LinkingError as e:
        return templates.TemplateResponse("settings.html", {
            "request": request, "actor": actor, "error": str(e),
            "active_bindings": [b for b in actor.bindings if b.revoked_at is None],
            "events": db.query(AuthEvent).filter(AuthEvent.actor_id == actor.actor_id).order_by(AuthEvent.created_at.desc()).limit(10).all()
        })

@app.post("/auth/link/start")
async def link_start(request: Request, actor: Actor = Depends(get_current_actor)):
    # In a real implementation, this would redirect to Authentik or another provider with a state token
    # For this demo, we simulate linking a new mock passkey provider
    return HTMLResponse("<body><p>Simulating external provider link...</p><form action='/auth/link/callback' method='POST'><button type='submit'>Complete Link</button></form></body>")

@app.post("/auth/link/callback")
async def link_callback(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    svc = LinkingService(db)
    try:
        svc.resolve_login_identity(
            provider_type=ProviderType.oauth_oidc,
            provider_name="google_oidc_preview",
            external_subject="mock_google_sub_789",
            external_email="linked@example.com",
            current_actor_id=actor.actor_id,
            request_ip=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
        return RedirectResponse(url="/account/settings", status_code=302)
    except LinkingError as e:
        return HTMLResponse(f"Linking failed: {e}", status_code=400)

@app.post("/account/passkey/enroll/start")
async def passkey_enroll_start(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    # Simulate passkey enrollment
    svc = LinkingService(db)
    svc.resolve_login_identity(
        provider_type=ProviderType.passkey,
        provider_name="webauthn",
        external_subject=f"passkey_{os.urandom(4).hex()}",
        current_actor_id=actor.actor_id,
        request_ip=request.client.host
    )
    redirect_url = request.session.pop("pending_hydra_redirect", "/account/settings")
    return RedirectResponse(url=redirect_url, status_code=302)


# ==========================================
# Web UI: Public Pages
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def landing_get(request: Request):
    """Public landing page explaining the Maestro."""
    actor_id = request.session.get("actor_id")
    actor = None
    if actor_id:
        db = SessionLocal()
        try:
            actor = db.query(Actor).filter(Actor.actor_id == actor_id).first()
        finally:
            db.close()
    
    return templates.TemplateResponse("landing.html", {"request": request, "actor": actor})

@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works_get(request: Request):
    return templates.TemplateResponse("how_it_works.html", {"request": request})

@app.get("/transparency", response_class=HTMLResponse)
async def transparency_get(request: Request):
    return templates.TemplateResponse("transparency.html", {"request": request})

@app.get("/safety", response_class=HTMLResponse)
async def safety_get(request: Request):
    return templates.TemplateResponse("safety.html", {"request": request})

# ==========================================
# Web UI: Maestro Platform (Human Actor)
# ==========================================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    svc = MaestroDomainService(db)
    try:
        data = svc.get_human_dashboard_data(actor.actor_id)
        opportunities = svc.get_open_opportunities(actor.actor_id)
    except MaestroDomainError:
        data = {"total_contributions": 0, "pending_settlements": 0, "settled_contributions": 0, "settled_total": 0}
        opportunities = []
        
    csrf_token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "actor": actor,
        "profile": getattr(actor, "human_profile", None),
        "stats": data,
        "opportunities": opportunities,
        "csrf_token": csrf_token
    })

@app.get("/compensation", response_class=HTMLResponse)
async def compensation_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    svc = MaestroDomainService(db)
    data = svc.get_compensation_data(actor.actor_id)
    return templates.TemplateResponse("compensation.html", {
        "request": request,
        "actor": actor,
        **data
    })

@app.get("/provenance/{contribution_id}", response_class=HTMLResponse)
async def provenance_receipt_get(contribution_id: str, request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    contrib = db.query(ExperienceContribution).filter_by(contribution_id=contribution_id, actor_id=actor.actor_id).first()
    if not contrib:
        raise HTTPException(status_code=404, detail="Contribution not found")
    
    prov = db.query(ProvenanceRecord).filter_by(contribution_id=contribution_id).first()
    return templates.TemplateResponse("provenance_receipt.html", {
        "request": request,
        "actor": actor,
        "contribution": contrib,
        "provenance": prov
    })

@app.post("/contributions/{contribution_id}/withdraw")
async def withdraw_contribution(request: Request, contribution_id: str, csrf_token: str = Form(...), actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    contrib = db.query(ExperienceContribution).filter_by(contribution_id=contribution_id, actor_id=actor.actor_id).first()
    if not contrib:
        raise HTTPException(status_code=404, detail="Contribution not found")

    contrib.withdrawn_at = datetime.now(timezone.utc)
    contrib.contribution_status = ContributionStatus.withdrawn
    db.commit()
    return RedirectResponse(url="/history", status_code=302)

@app.get("/consent/settings", response_class=HTMLResponse)
async def consent_settings_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    profile = actor.human_profile
    scope = None
    if profile and profile.consent_profile_id:
        scope = db.query(ConsentScope).filter_by(consent_scope_id=profile.consent_profile_id).first()

    csrf_token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("consent.html", {
        "request": request,
        "actor": actor,
        "scope": scope,
        "csrf_token": csrf_token
    })

@app.post("/consent/settings", response_class=HTMLResponse)
async def consent_settings_post(
    request: Request,
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db)
):
    form_data = await request.form()
    csrf_token = form_data.get("csrf_token", "")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    classes = form_data.getlist("classes")
    modalities_list = form_data.getlist("modalities")
    retention = form_data.get("retention")
    visibility = form_data.get("visibility")
    auto_revoke = form_data.get("auto_revoke") == "true"
    
    modality_permissions = {mod: True for mod in modalities_list}
    
    svc = MaestroDomainService(db)
    try:
        scope = svc.update_consent_scope(actor.actor_id, classes, modality_permissions)
        scope.retention_preferences = {"period": retention}
        scope.visibility_policy = {"level": visibility}
        scope.revocation_policy = {"auto_revoke": auto_revoke}
        db.commit()
        
        return templates.TemplateResponse("consent.html", {
            "request": request,
            "actor": actor,
            "scope": scope,
            "message": "Consent scope updated successfully."
        })
    except MaestroDomainError as e:
        return templates.TemplateResponse("consent.html", {
            "request": request,
            "actor": actor,
            "scope": None,
            "error": str(e)
        })

@app.get("/history", response_class=HTMLResponse)
async def history_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    contributions = db.query(ExperienceContribution).filter_by(actor_id=actor.actor_id).order_by(ExperienceContribution.submitted_at.desc()).all()

    csrf_token = secrets.token_urlsafe(32)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("history.html", {
        "request": request,
        "actor": actor,
        "contributions": contributions,
        "csrf_token": csrf_token
    })

# ==========================================
# Domain Specific APIs (Protected)
# ==========================================

class GroundingRequestCreate(BaseModel):
    title: str
    description: str
    request_class: str
    hypothesis_entropy_reduction: float
    hypothesis_phase_sync: Optional[str] = None
    success_signal_type: Optional[str] = None
    compensation_budget: float
    required_contributions: int = 50

@app.post("/agent/requests")
async def create_grounding_request(
    req: GroundingRequestCreate, 
    token_data: dict = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    """Agent creates a grounding request. The token 'sub' is the agent client_id."""
    agent_client_id = token_data.get("sub")
    # Resolve the agent actor
    binding = db.query(CredentialBinding).filter_by(external_subject=agent_client_id, provider_type=ProviderType.client_credentials).first()
    if not binding:
        raise HTTPException(status_code=403, detail="Agent identity not found")
        
    svc = MaestroDomainService(db)
    request_record = svc.create_grounding_request(
        agent_id=binding.actor_id,
        title=req.title,
        description=req.description,
        request_class=req.request_class,
        entropy_target=req.hypothesis_entropy_reduction,
        budget=req.compensation_budget,
        required_contribs=req.required_contributions
    )
    # Ensure hypothesis fields are also saved
    request_record.hypothesis_phase_sync = req.hypothesis_phase_sync
    request_record.success_signal_type = req.success_signal_type
    db.commit()
    
    return {"status": "success", "request_id": request_record.request_id}

@app.post("/viability/requests/evaluate/{request_id}")
async def evaluate_request_viability(
    request_id: str,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Agent requests viability evaluation for a grounding request."""
    svc = MaestroDomainService(db)
    try:
        eval_record = svc.evaluate_viability(request_id)
        return {
            "evaluation_id": eval_record.evaluation_id,
            "final_decision": eval_record.final_decision,
            "explanation": eval_record.explanation
        }
    except MaestroDomainError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/viability/contributions/evaluate/{contribution_id}")
async def evaluate_contribution_viability(
    contribution_id: str,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Evaluate if a contribution is viable."""
    svc = MaestroDomainService(db)
    try:
        eval_record = svc.evaluate_contribution_viability(contribution_id)
        return {
            "evaluation_id": eval_record.evaluation_id,
            "final_decision": eval_record.final_decision,
            "explanation": eval_record.explanation
        }
    except MaestroDomainError as e:
        raise HTTPException(status_code=400, detail=str(e))

class SettlementEvaluateRequest(BaseModel):
    evidence_metadata: dict
    confidence: float

@app.post("/viability/settlement/evaluate/{request_id}")
async def evaluate_settlement_viability(
    request_id: str,
    req: SettlementEvaluateRequest,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Evaluate if a settlement is viable based on entropy and phase proofs."""
    svc = MaestroDomainService(db)
    try:
        eligible, explanation = svc.evaluate_settlement_viability(request_id, req.evidence_metadata, req.confidence)
        return {
            "eligible": eligible,
            "explanation": explanation
        }
    except MaestroDomainError as e:
        raise HTTPException(status_code=400, detail=str(e))

class AgentRegisterRequest(BaseModel):
    owner_id: str
    client_id: str
    software_name: str = "Maestro Agent SDK"

@app.post("/domain/agent/register")
async def register_agent(req: AgentRegisterRequest, db: Session = Depends(get_db)):
    """Register a new agent in the domain database."""
    # Ensure owner exists
    owner = db.query(Actor).filter(Actor.actor_id == req.owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner actor not found")

    # Create Agent Actor
    agent_actor = Actor(actor_type=ActorType.agent, display_name=f"Agent {req.client_id}")
    db.add(agent_actor)
    db.flush()

    # Create Agent Profile
    profile = AgentProfile(
        actor_id=agent_actor.actor_id,
        software_name=req.software_name,
        operator_actor_id=owner.actor_id
    )
    db.add(profile)

    # Bind Client Credentials
    binding = CredentialBinding(
        actor_id=agent_actor.actor_id,
        provider_type=ProviderType.client_credentials,
        provider_name="ory_hydra",
        external_subject=req.client_id
    )
    db.add(binding)
    db.commit()
    
    return {"status": "success", "agent_actor_id": agent_actor.actor_id}

@app.get("/agent/dashboard", response_class=HTMLResponse)
async def agent_dashboard_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    if actor.actor_type != ActorType.agent:
        raise HTTPException(status_code=403, detail="Must be an agent actor")
    
    requests = db.query(GroundingRequest).filter_by(agent_actor_id=actor.actor_id).all()
    
    return templates.TemplateResponse("agent_dashboard.html", {
        "request": request,
        "actor": actor,
        "requests": requests
    })

@app.get("/operator/review-queue", response_class=HTMLResponse)
async def operator_review_get(request: Request, actor: Actor = Depends(get_current_actor), db: Session = Depends(get_db)):
    if actor.actor_type != ActorType.operator:
        raise HTTPException(status_code=403, detail="Must be an operator")
        
    evaluations = db.query(ViabilityEvaluation).filter_by(final_decision=ViabilityDecision.review).all()
    
    return templates.TemplateResponse("operator_review.html", {
        "request": request,
        "actor": actor,
        "evaluations": evaluations
    })
@app.get("/auth/me")
async def get_auth_me(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Fetch current authenticated profile."""
    actor_id = token_data.get("sub")
    actor = db.query(Actor).filter(Actor.actor_id == actor_id).first()
    if not actor:
        raise HTTPException(status_code=404, detail="Actor not found")
    
    return {
        "actor_id": actor.actor_id,
        "type": actor.actor_type,
        "display_name": actor.display_name,
        "assurance_level": actor.assurance_level
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





