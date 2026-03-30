import os
import pytest
import sys
from datetime import datetime

# Set TESTING environment variable BEFORE importing database/models
os.environ["TESTING"] = "true"

# Add current dir to sys.path to resolve local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import engine, SessionLocal, Base
from models import Actor, HumanProfile, CredentialBinding, ActorType, ProviderType, AuthEventType
from services import LinkingService
from exceptions import LinkingError

@pytest.fixture(scope="function")
def db():
    # Create tables
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop tables for clean state between tests
        Base.metadata.drop_all(bind=engine)

def test_new_user_creation(db):
    svc = LinkingService(db)
    provider_name = "google"
    external_sub = "google_123"
    email = "newuser@example.com"
    
    actor, is_new = svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name=provider_name,
        external_subject=external_sub,
        external_email=email
    )
    
    assert is_new is True
    assert actor.actor_type == ActorType.human
    assert actor.display_name == "newuser"
    
    # Verify DB records
    db_actor = db.query(Actor).filter_by(actor_id=actor.actor_id).first()
    assert db_actor is not None
    assert db_actor.human_profile.email == email
    assert len(db_actor.bindings) == 1
    assert db_actor.bindings[0].external_subject == external_sub

def test_existing_user_login(db):
    svc = LinkingService(db)
    provider_name = "google"
    external_sub = "google_123"
    
    # First login
    actor1, is_new1 = svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name=provider_name,
        external_subject=external_sub,
        external_email="alice@test.com"
    )
    initial_seen = actor1.last_seen_at
    
    # Second login
    actor2, is_new2 = svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name=provider_name,
        external_subject=external_sub
    )
    
    assert is_new2 is False
    assert actor1.actor_id == actor2.actor_id
    assert actor2.last_seen_at > initial_seen

def test_link_new_provider_to_existing_actor(db):
    svc = LinkingService(db)
    
    # Create initial actor
    actor, _ = svc.resolve_login_identity(
        provider_type=ProviderType.email_link,
        provider_name="magic_link",
        external_subject="alice@test.com",
        external_email="alice@test.com"
    )
    
    # Link a second provider
    actor2, is_new = svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name="github",
        external_subject="github_alice",
        current_actor_id=actor.actor_id
    )
    
    assert is_new is False
    assert actor.actor_id == actor2.actor_id
    assert len(actor.bindings) == 2
    assert any(b.provider_name == "github" for b in actor.bindings)

def test_email_collision_rejected(db):
    svc = LinkingService(db)
    email = "collision@test.com"
    
    # Create user 1
    svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name="google",
        external_subject="google_user",
        external_email=email
    )
    
    # Attempt login with user 2 having same email but different subject/provider
    with pytest.raises(LinkingError) as excinfo:
        svc.resolve_login_identity(
            provider_type=ProviderType.oauth_oidc,
            provider_name="github",
            external_subject="github_user",
            external_email=email
        )
    assert "already exists" in str(excinfo.value)

def test_unlink_provider(db):
    svc = LinkingService(db)
    
    # Create actor with 2 bindings
    actor, _ = svc.resolve_login_identity(
        provider_type=ProviderType.email_link,
        provider_name="magic_link",
        external_subject="alice@test.com",
        external_email="alice@test.com"
    )
    svc.resolve_login_identity(
        provider_type=ProviderType.oauth_oidc,
        provider_name="github",
        external_subject="github_alice",
        current_actor_id=actor.actor_id
    )
    
    binding_to_unlink = actor.bindings[0].binding_id
    
    # Unlink
    success = svc.unlink_provider(actor.actor_id, binding_id=binding_to_unlink)
    assert success is True
    
    db.refresh(actor)
    unlinked = next(b for b in actor.bindings if b.binding_id == binding_to_unlink)
    assert unlinked.revoked_at is not None

def test_unlink_last_provider_rejected(db):
    svc = LinkingService(db)
    
    actor, _ = svc.resolve_login_identity(
        provider_type=ProviderType.email_link,
        provider_name="magic_link",
        external_subject="alice@test.com",
        external_email="alice@test.com"
    )
    
    binding_id = actor.bindings[0].binding_id
    
    with pytest.raises(LinkingError) as excinfo:
        svc.unlink_provider(actor.actor_id, binding_id=binding_id)
    assert "last remaining credential" in str(excinfo.value)
