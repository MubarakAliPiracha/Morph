import { BACKEND_URL } from '@/lib/config';
import type { WorldObject } from '@/lib/world';

export type JointInfo = {
  index: number;
  name: string;
  type: 'revolute' | 'prismatic' | 'continuous' | 'fixed' | string;
  lower_limit: number | null;
  upper_limit: number | null;
};

export type SensorInfo = {
  name: string;
  type: string;
  link: string;
  topic: string;
  rate_hz?: number;
  range_m?: [number, number];
  samples?: number;
};

export type RobotInfo = {
  source: string;
  name: string;
  urdf_url: string;
  root_url: string;
  joints: JointInfo[];
  mobile: boolean;
  wheels: string[];
  sensors?: SensorInfo[];
  scale?: { unit_m: number; spawn_x: number; factor: number };
  warnings?: string[];
  upload_id?: string;
};

export type Health = { ok: boolean; ollama: boolean; model: string; robot: RobotInfo };

export type CommandResult = {
  ok: boolean;
  source: string;
  reply: string | null;
  plan: Record<string, unknown>[];
  repeat: boolean;
  world: WorldObject[] | null;
  warnings: string[];
  llm: { provider: string | null; model: string | null };
  llm_error: string | null;
  path?: [number, number][];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BACKEND_URL + path, init);
  } catch {
    throw new Error('Backend is offline. Start it with "npm run backend".');
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>('/api/health'),
  selectRobot: (source: string) => request<RobotInfo>('/api/robot/select', json({ source })),
  activateRobot: (source: string) =>
    request<{ ok: boolean; source: string; rebooting: boolean }>(
      '/api/robot/activate',
      json({ source }),
    ),
  uploadRobot: (files: File[]) => {
    const form = new FormData();
    for (const f of files) {
      const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath;
      form.append('files', f, rel || f.name);
    }
    return request<RobotInfo>('/api/robot/upload', { method: 'POST', body: form });
  },
  resetRobot: () => request<RobotInfo>('/api/robot/reset', { method: 'POST' }),
  command: (text: string, mode: 'robot' | 'map' = 'robot') => request<CommandResult>('/api/command', json({ text, mode })),
  setWorld: (objects: WorldObject[]) => request<{ ok: boolean }>('/api/world', { ...json({ objects }), method: 'PUT' }),
  stop: () => request<{ ok: boolean }>('/api/stop', { method: 'POST' }),
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Wait out an activation reboot and return whatever robot the sim came back with.
 *
 * Two phases on purpose: right after activate, the OLD server still answers for a
 * second, so "source doesn't match yet" means nothing. Only an answer that either
 * matches the requested robot or arrives after observed downtime is the new boot —
 * a non-matching robot then means the entrypoint fell back (broken upload).
 */
export async function waitForRobot(source: string, timeoutMs = 240_000): Promise<RobotInfo> {
  const start = Date.now();
  let sawDown = false;
  while (Date.now() - start < timeoutMs) {
    await sleep(3000);
    try {
      const health = await api.health();
      if (health.robot.source === source) return health.robot;
      if (sawDown) return health.robot;
    } catch {
      sawDown = true;
    }
  }
  throw new Error('The simulator did not come back after the reboot. Check "docker compose logs sim".');
}
