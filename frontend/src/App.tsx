import { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

type Evidence = { source_id: number; chunk_id: string; quote: string };
type QAResponse = {
  answer: string;
  evidence?: Evidence[];
  object_summary?: any;
  sources?: any[];
  plan?: any;
  object_checks?: any[];
};

const DEFAULT_EMAIL = "test@example.com";
const DEFAULT_PASSWORD = "123456";

export default function App() {
  const API_BASE = useMemo(() => {
    return (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";
  }, []);

  const [tab, setTab] = useState<"auth" | "session" | "qa">("auth");

  const [email, setEmail] = useState(DEFAULT_EMAIL);
  const [password, setPassword] = useState(DEFAULT_PASSWORD);

  const [token, setToken] = useState<string>(() => localStorage.getItem("token") || "");

  const [objectsJson, setObjectsJson] = useState<string>(() => `{
  "object_list": [
    {"type":"LINE","layer":"Highway","start":[0,0],"end":[10,0]},
    {"type":"POLYLINE","layer":"Windows","points":[[1,1],[2,1]],"closed":false},
    {"type":"POLYLINE","layer":"Windows","points":[[3,1],[4,1]],"closed":false}
  ]
}`);

  const [question, setQuestion] = useState<string>(
    "On page 14, what is the restriction about the principal elevation and highway? 1-2 sentences."
  );

  const [topK, setTopK] = useState<number>(5);

  const [status, setStatus] = useState<string>("");
  const [qaResp, setQaResp] = useState<QAResponse | null>(null);

  const authHeaders = useMemo(() => {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }, [token]);

  async function register() {
    setStatus("Registering...");
    setQaResp(null);
    try {
      const r = await axios.post(`${API_BASE}/auth/register`, { email, password });
      setStatus(`✅ Registered: ${r.data.email} (${r.data.user_id})`);
    } catch (e: any) {
      setStatus(`❌ Register failed: ${readErr(e)}`);
    }
  }

  async function login() {
    setStatus("Logging in...");
    setQaResp(null);
    try {
      const r = await axios.post(`${API_BASE}/auth/login`, { email, password });
      const t = r.data?.access_token || "";
      setToken(t);
      localStorage.setItem("token", t);
      setStatus("✅ Logged in. Token saved.");
      setTab("session");
    } catch (e: any) {
      setStatus(`❌ Login failed: ${readErr(e)}`);
    }
  }

  async function logout() {
    setToken("");
    localStorage.removeItem("token");
    setStatus("Logged out.");
    setQaResp(null);
  }

  async function updateObjects() {
    setStatus("Updating objects...");
    setQaResp(null);
    try {
      const parsed = JSON.parse(objectsJson);
      const r = await axios.put(`${API_BASE}/session/objects`, parsed, { headers: authHeaders });
      setStatus(`✅ Objects updated. Count=${r.data.object_count}`);
      setTab("qa");
    } catch (e: any) {
      setStatus(`❌ Update objects failed: ${readErr(e)}`);
    }
  }

  async function ask() {
    setStatus("Asking...");
    setQaResp(null);
    try {
      const r = await axios.post(
        `${API_BASE}/qa`,
        { question, top_k: topK },
        { headers: { ...authHeaders, "Content-Type": "application/json" } }
      );
      setQaResp(r.data);
      setStatus("✅ Answer received.");
    } catch (e: any) {
      setStatus(`❌ QA failed: ${readErr(e)}`);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">AICI Hybrid RAG Demo</div>
          <div className="subtitle">Backend (JWT + session) → Agent (Hybrid RAG + verification)</div>
        </div>
        <div className="pillrow">
          <span className={`pill ${token ? "pill-ok" : "pill-warn"}`}>
            {token ? "Authenticated" : "Not authenticated"}
          </span>
          <span className="pill">API: {API_BASE}</span>
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "auth" ? "tab active" : "tab"} onClick={() => setTab("auth")}>
          1) Auth
        </button>
        <button className={tab === "session" ? "tab active" : "tab"} onClick={() => setTab("session")} disabled={!token}>
          2) Session Objects
        </button>
        <button className={tab === "qa" ? "tab active" : "tab"} onClick={() => setTab("qa")} disabled={!token}>
          3) Q&A
        </button>
      </nav>

      <main className="grid">
        {tab === "auth" && (
          <section className="card">
            <h2>Authentication</h2>
            <p className="muted">Register then login to get a JWT token. Token is stored in localStorage.</p>

            <div className="row">
              <label>Email</label>
              <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="test@example.com" />
            </div>
            <div className="row">
              <label>Password</label>
              <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="min 6 chars" />
            </div>

            <div className="actions">
              <button className="btn" onClick={register}>Register</button>
              <button className="btn primary" onClick={login}>Login</button>
              <button className="btn ghost" onClick={logout} disabled={!token}>Logout</button>
            </div>
          </section>
        )}

        {tab === "session" && (
          <section className="card">
            <h2>Session Object List (Ephemeral)</h2>
            <p className="muted">
              Edit JSON and click <b>Update Objects</b>. Backend stores it in-memory per user session.
            </p>
            <textarea value={objectsJson} onChange={(e) => setObjectsJson(e.target.value)} spellCheck={false} />
            <div className="actions">
              <button className="btn primary" onClick={updateObjects}>Update Objects</button>
            </div>
          </section>
        )}

        {tab === "qa" && (
          <section className="card">
            <h2>Ask a Question</h2>
            <p className="muted">The agent combines retrieved PDF chunks + current session objects.</p>

            <div className="row">
              <label>Question</label>
              <input value={question} onChange={(e) => setQuestion(e.target.value)} />
            </div>

            <div className="row">
              <label>top_k</label>
              <input
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                type="number"
                min={1}
                max={10}
              />
            </div>

            <div className="actions">
              <button className="btn primary" onClick={ask}>Ask</button>
            </div>

            {qaResp && (
              <div className="result">
                <div className="resultTitle">Answer</div>
                <div className="answer">{qaResp.answer}</div>

                <div className="resultTitle">Evidence</div>
                {(qaResp.evidence || []).length === 0 ? (
                  <div className="muted">No evidence returned (may be a direct strategy).</div>
                ) : (
                  <ul className="evidence">
                    {(qaResp.evidence || []).map((ev, i) => (
                      <li key={i}>
                        <div className="evMeta">
                          source_id={ev.source_id} · chunk_id={ev.chunk_id}
                        </div>
                        <div className="quote">“{ev.quote}”</div>
                      </li>
                    ))}
                  </ul>
                )}

                <details>
                  <summary>Debug (plan, object_summary, sources)</summary>
                  <pre>{JSON.stringify({ plan: qaResp.plan, object_summary: qaResp.object_summary, sources: qaResp.sources }, null, 2)}</pre>
                </details>
              </div>
            )}
          </section>
        )}

        <section className="card">
          <h2>Status</h2>
          <div className="status">{status || "—"}</div>
          <div className="muted small">
            Tip: After restarting backend, in-memory session resets. Re-login + update objects.
          </div>
        </section>
      </main>

      <footer className="footer muted small">
        Minimal UI for the challenge. Focus: correctness of hybrid RAG + session state.
      </footer>
    </div>
  );
}

function readErr(e: any): string {
  const r = e?.response;
  if (r?.data) return typeof r.data === "string" ? r.data : JSON.stringify(r.data);
  return e?.message || String(e);
}
