import { apiFetch } from "./client";

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export function login(username: string, password: string): Promise<TokenPair> {
  return apiFetch<TokenPair>("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}
