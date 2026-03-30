"""
Bootstrap Script for Project Maestro Auth Subsystem

This script initializes the Ory Hydra authorization server with the
necessary OAuth2 clients for local development.
"""

import os
import requests
import time

HYDRA_ADMIN_URL = os.getenv("HYDRA_ADMIN_URL", "http://127.0.0.1:4445")
CLIENT_ID = os.getenv("APP_CLIENT_ID", "Maestro-local-client")
CLIENT_SECRET = os.getenv("APP_CLIENT_SECRET", "Maestro-local-secret")

def wait_for_hydra():
    """Wait for Hydra admin API to be available."""
    print("Waiting for Hydra Admin API to be healthy...")
    for _ in range(30):
        try:
            resp = requests.get(f"{HYDRA_ADMIN_URL}/health/ready")
            if resp.status_code == 200:
                print("Hydra is ready!")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def create_agent_client():
    """Create the OAuth2 client used by AI Agents for client credentials."""
    print(f"Provisioning OAuth2 client '{CLIENT_ID}' in Hydra...")
    
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_types": ["client_credentials", "authorization_code", "refresh_token"],
        "response_types": ["token", "code", "id_token"],
        "scope": "offline_access write:studies read:profile admin",
        "audience": ["Maestro-api"],
        "token_endpoint_auth_method": "client_secret_post"
    }
    
    # Check if exists
    try:
        resp = requests.get(f"{HYDRA_ADMIN_URL}/admin/clients/{CLIENT_ID}")
        if resp.status_code == 200:
            print("Client already exists. Updating...")
            resp = requests.put(f"{HYDRA_ADMIN_URL}/admin/clients/{CLIENT_ID}", json=payload)
            resp.raise_for_status()
            print("Client updated successfully.")
            return
    except requests.RequestException:
        pass # Client doesn't exist or hydra not available

    # Create new
    try:
        resp = requests.post(f"{HYDRA_ADMIN_URL}/admin/clients", json=payload)
        resp.raise_for_status()
        print("Client created successfully.")
    except requests.RequestException as e:
        print(f"Failed to provision client: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response: {e.response.text}")

if __name__ == "__main__":
    if wait_for_hydra():
        create_agent_client()
    else:
        print("Error: Hydra failed to become ready.")
        exit(1)

