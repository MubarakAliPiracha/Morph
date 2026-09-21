'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useRef, useState, type DragEvent, type FormEvent } from 'react';
import {
  AlertTriangle,
  Bot,
  ChevronLeft,
  FolderUp,
  Loader2,
  Map as MapIcon,
  OctagonX,
  Radar,
  RotateCcw,
  Send,
  Shapes,
  Trash2,
  Upload,
  Wand2,
  Terminal,
} from 'lucide-react';
import { Logo } from '@/components/logo';
import { SceneViewport } from '@/components/scene-viewport';
import { ShapesPanel } from '@/components/shapes-panel';
import { ViewportToolbar, type Tool } from '@/components/viewport-toolbar';
import { api, waitForRobot, type CommandResult, type Health, type RobotInfo } from '@/lib/api';
import { useSims } from '@/lib/sims';
import { useRobotSocket } from '@/lib/use-robot-socket';
import { ConsolePanel } from '@/components/console-panel';
import { PlanSteps, type PlanStep } from '@/components/plan-steps';
import { SHAPES, newObject, type WorldObject } from '@/lib/world';

type Msg = {
  id: number;
  text: string;
  status: 'pending' | 'ok' | 'error';
  result?: CommandResult;
  error?: string;
};

const UPLOAD_ACCEPT = '.urdf,.xacro,.zip,.stl,.obj,.dae,.glb,.gltf,.mtl,.png,.jpg,.jpeg';

const MAP_SUGGESTIONS = [
  'Build a maze',
  'Make an obstacle course',
  'Build a small room with tables and shelves',
  'Create a warehouse with crates and aisles',
  'Build a basketball court',
];
const ARM_SUGGESTIONS = ['Flap for 3 seconds', 'Move in a circle for 4 seconds'];
const ROVER_SUGGESTIONS = [
  'Drive forward until you see the wall, then turn left',
  'Drive 2 meters, turn right 90 degrees, drive 1 meter',
  'Avoid obstacles',
];

export function SimWorkspace() {
  const { id } = useParams<{ id: string }>();
  const { sims, ready, update } = useSims();
  const sim = sims.find((s) => s.id === id);
  const simRef = useRef(sim);
  simRef.current = sim;

  const { latest, snapshot, connected } = useRobotSocket();
  const [robot, setRobot] = useState<RobotInfo | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [activePlan, setActivePlan] = useState<PlanStep[]>([]);
  const [dragging, setDragging] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [mapMessages, setMapMessages] = useState<Msg[]>([]);
  const [panel, setPanel] = useState<'robot' | 'map' | 'console'>('robot');
  const [traceKey, setTraceKey] = useState(0); // bump to clear the robot's path trail
  const [tool, setTool] = useState<Tool>('select');
  const [follow, setFollow] = useState(false);
  const [viewKey, setViewKey] = useState(0); // bump to reset the camera
  const [plannedPath, setPlannedPath] = useState<[number, number][] | null>(null);
  const [draft, setDraft] = useState('');
  const [tab, setTab] = useState<'robot' | 'shapes'>('robot');
  const [objects, setObjects] = useState<WorldObject[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const worldLoaded = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const logEnd = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  // Poll backend status (only surfaced when it goes offline).
  useEffect(() => {
    let alive = true;
    const poll = () =>
      api
        .health()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth(null));
    poll();
    const t = setInterval(poll, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // Restore this simulation's saved world once.
  useEffect(() => {
    if (!ready || !sim || worldLoaded.current) return;
    setObjects(sim.world ?? []);
    worldLoaded.current = true;
  }, [ready, sim]);

  // Persist the world locally.
  useEffect(() => {
    if (!worldLoaded.current) return;
    const t = setTimeout(() => update(id, { world: objects }), 300);
    return () => clearTimeout(t);
  }, [objects, id, update]);

  // Keep the backend physics world in step: on every edit, and again after the robot (re)loads.
  useEffect(() => {
    if (!connected || !robot || !worldLoaded.current) return;
    const t = setTimeout(() => void api.setWorld(objects).catch((e: Error) => setNotice(e.message)), 120);
    return () => clearTimeout(t);
  }, [objects, connected, robot]);

  // (Re)load this simulation's robot into the backend whenever the socket connects.
  useEffect(() => {
    if (!connected || !ready || !simRef.current) return;
    let cancelled = false;
    api
      .selectRobot(simRef.current.robotSource)
      .then((r) => !cancelled && setRobot(r))
      .catch(async (e: Error) => {
        try {
          const r = await api.selectRobot('default');
          if (cancelled) return;
          setRobot(r);
          setNotice(`${e.message} Loaded the default arm instead.`);
        } catch (e2) {
          if (!cancelled) setNotice((e2 as Error).message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [connected, ready, sim?.id]);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, mapMessages, panel]);

  const adoptRobot = useCallback(
    (r: RobotInfo) => {
      setRobot(r);
      if (r.warnings?.length) setNotice(r.warnings.join(' '));
      update(id, { robotSource: r.source, robotName: r.name });
    },
    [id, update],
  );

  const upload = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setBusy(true);
      setNotice(null);
      try {
        const converted = await api.uploadRobot(files);
        if (!converted.upload_id) {
          adoptRobot(converted);
          return;
        }
        // The sim boots one robot per container, so activation is a reboot: the
        // backend goes down and comes back up running the uploaded robot.
        setNotice(
          `${converted.warnings?.[0] ?? 'Converted.'} Rebooting the simulator with your robot — about a minute…`,
        );
        await api.activateRobot(converted.upload_id);
        const active = await waitForRobot(converted.upload_id);
        adoptRobot(active);
        setNotice(
          active.source === converted.upload_id
            ? `${active.name} is live — this is your uploaded robot. Try "drive forward 1 meter".`
            : 'Your robot converted but failed to boot; the simulator reverted to the TurtleBot. Check "docker compose logs sim".',
        );
      } catch (e) {
        setNotice((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [adoptRobot],
  );

  const reset = async () => {
    try {
      setRobot(await api.resetRobot());
      setTraceKey((k) => k + 1);
      setPlannedPath(null);
      setNotice(null);
    } catch (e) {
      setNotice((e as Error).message);
    }
  };

  const send = async (text: string, mode: 'robot' | 'map' = panel === 'map' ? 'map' : 'robot') => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const setList = mode === 'map' ? setMapMessages : setMessages;
    if (mode === 'robot') {
      setTraceKey((k) => k + 1); // new request: start a fresh trail and drop the old route
      setPlannedPath(null);
    }
    const msgId = nextId.current++;
    setList((m) => [...m, { id: msgId, text: trimmed, status: 'pending' }]);
    setDraft('');
    try {
      const result = await api.command(trimmed, mode);
      if (result.world) {
        setObjects(result.world);
        setSelectedId(null);
      }
      if (mode === 'robot') {
        setPlannedPath(result.path && result.path.length > 1 ? result.path : null);
        if (result.plan?.length) setActivePlan(result.plan as unknown as PlanStep[]);
      }
      setList((m) => m.map((x) => (x.id === msgId ? { ...x, status: 'ok', result } : x)));
      if (mode === 'robot' && sim && !sim.named) {
        update(id, { name: `tb3 · ${trimmed.slice(0, 30)}`, named: true });
      } else {
        update(id, {});
      }
    } catch (e) {
      setList((m) => m.map((x) => (x.id === msgId ? { ...x, status: 'error', error: (e as Error).message } : x)));
    }
  };

  const runDemo = () => {
    // Stage the classic demo without typing: a wall ahead, then the wall-follow command.
    const wallDef = SHAPES.find((s) => s.preset === 'wall') ?? SHAPES[0];
    if (!objects.length) {
      const wall = newObject(wallDef, objects, robot?.scale);
      setObjects((prev) => [...prev, wall]);
    }
    // Small delay so the debounced world sync reaches the sim before the plan runs.
    setTimeout(() => void send('Drive forward until you see the wall, then turn left', 'robot'), 600);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(draft);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void upload(Array.from(e.dataTransfer.files));
  };

  // World edits
  const addObject = (o: WorldObject) => {
    setObjects((prev) => [...prev, o]);
    setSelectedId(o.id);
  };
  const changeObject = (o: WorldObject) => setObjects((prev) => prev.map((x) => (x.id === o.id ? o : x)));
  const deleteObject = (oid: string) => {
    setObjects((prev) => prev.filter((x) => x.id !== oid));
    setSelectedId(null);
  };
  const moveObject = useCallback(
    (oid: string, x: number, y: number) => setObjects((prev) => prev.map((o) => (o.id === oid ? { ...o, x, y } : o))),
    [],
  );
  const transformObject = useCallback(
    (oid: string, patch: Partial<Pick<WorldObject, 'x' | 'y' | 'z' | 'yaw'>>) =>
      setObjects((prev) => prev.map((o) => (o.id === oid ? { ...o, ...patch } : o))),
    [],
  );
  const selectObject = useCallback((oid: string | null) => {
    setSelectedId(oid);
    if (oid) setTab('shapes');
  }, []);

  // Example chips land here with a command attached: run it once, then clear it
  // so a reload never replays it.
  const pendingRan = useRef(false);
  useEffect(() => {
    const cmd = sim?.pendingCommand;
    if (!cmd || !health || !robot || pendingRan.current) return;
    pendingRan.current = true;
    update(id, { pendingCommand: undefined });
    // Deliberately NOT cleaned up on dep change: update() above mutates `sim`,
    // which re-runs this effect immediately - a cleanup would cancel the timer
    // before it ever fired. pendingRan guards against double-sends instead.
    setTimeout(() => void send(cmd, 'robot'), 900); // let the world sync land first
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sim?.pendingCommand, health, robot]);

  // CAD-style shortcuts (ignored while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      // Q/W/E for Figma/Blender muscle memory; V/M/R kept as aliases.
      const map: Record<string, Tool> = {
        q: 'select', w: 'move', e: 'rotate',
        v: 'select', m: 'move', r: 'rotate',
        h: 'pan', o: 'orbit',
      };
      if (map[k]) setTool(map[k]);
      else if (k === 'f') setViewKey((n) => n + 1);
      else if (k === '.') setFollow((prev) => !prev);
      else if (k === 'escape') {
        setSelectedId(null);
        setTool('select');
      } else if ((k === 'delete' || k === 'backspace') && selectedId) {
        setObjects((prev) => prev.filter((o) => o.id !== selectedId));
        setSelectedId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedId]);

  if (ready && !sim) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-app text-fg">
        <p>This simulation doesn&apos;t exist in this browser.</p>
        <Link href="/" className="rounded-full bg-brand px-5 py-2 font-semibold text-white">
          Back to home
        </Link>
      </div>
    );
  }

  const movable = robot?.joints.filter((j) => j.type !== 'fixed') ?? [];
  const mobile = !!robot?.mobile;
  const suggestions = mobile
    ? ROVER_SUGGESTIONS
    : [...ARM_SUGGESTIONS, ...(movable[0] ? [`Rotate ${movable[0].name} by 45 degrees`] : [])];
  const list = panel === 'map' ? mapMessages : messages;
  const running = snapshot.active;
  const front = snapshot.sensor?.front;

  return (
    <div className="flex h-screen flex-col bg-app text-fg">
      <header className="topbar z-20 flex h-16 shrink-0 items-center justify-between gap-4 px-4 shadow-md">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/" aria-label="Back to home" className="rounded-full p-1.5 text-white/90 hover:bg-white/15">
            <ChevronLeft size={22} />
          </Link>
          <div className="hidden sm:block">
            <Logo />
          </div>
          <input
            value={sim?.name ?? ''}
            onChange={(e) => update(id, { name: e.target.value })}
            aria-label="Simulation name"
            className="w-44 min-w-0 rounded-lg border border-white/25 bg-white/10 px-3 py-1.5 text-sm font-semibold text-white outline-none placeholder:text-white/60 focus:bg-white/20 sm:w-64"
          />
        </div>

        <div className="flex min-w-0 shrink items-center gap-2">
          <div
            role="status"
            className={`flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-bold ring-1 transition ${
              running ? 'bg-accent text-white ring-white/40' : busy ? 'bg-warn/80 text-black ring-warn/60' : 'bg-white/15 text-white/90 ring-white/25'
            }`}
          >
            <span className={`h-2.5 w-2.5 rounded-full ${running || busy ? 'animate-pulse bg-white' : 'bg-white/50'}`} />
            {running ? 'Running' : busy ? 'Planning' : 'Idle'}
          </div>
          <button
            onClick={() => void api.stop().catch((e: Error) => setNotice(e.message))}
            disabled={!running}
            className="flex items-center gap-1.5 rounded-full bg-pop px-3.5 py-1.5 text-sm font-bold text-white shadow transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:brightness-100"
          >
            <OctagonX size={16} />
            Stop
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-1.5 rounded-full bg-white/15 px-3.5 py-1.5 text-sm font-semibold text-white ring-1 ring-white/30 transition hover:bg-white/25"
          >
            <RotateCcw size={15} />
            Reset
          </button>
        </div>
      </header>

      {(!health || notice) && (
        <div className="flex items-center gap-2 bg-warn/20 px-4 py-2 text-sm text-fg">
          <AlertTriangle size={16} className="shrink-0 text-warn" />
          {notice ?? (
            <>
              Can&apos;t reach the simulator. Run <code className="rounded bg-surfaceAlt px-1.5 py-0.5 font-mono">npm run backend</code>{' '}
              (or <code className="rounded bg-surfaceAlt px-1.5 py-0.5 font-mono">npm run dev:all</code>) and this page will
              connect automatically.
            </>
          )}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_360px]">
        {/* Left: robot + shapes */}
        <aside className="order-2 flex min-h-0 flex-col border-line bg-surface lg:order-1 lg:border-r">
          <div className="grid shrink-0 grid-cols-2 border-b border-line">
            {(
              [
                ['robot', 'Robot', Bot],
                ['shapes', 'Shapes', Shapes],
              ] as const
            ).map(([key, label, Icon]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`flex items-center justify-center gap-2 border-b-2 px-3 py-3 text-sm font-bold transition ${
                  tab === key ? 'border-brand text-brand' : 'border-transparent text-muted hover:text-fg'
                }`}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === 'robot' ? (
              <div className="min-w-0 space-y-4">
                {/* 1. Active robot - the hero card. Everything else is quieter. */}
                <section className="rounded-xl border border-brand/50 bg-brand/5 p-3">
                  <div className="flex items-center gap-2 font-semibold text-brand">
                    <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                    <span className="truncate">{robot?.name ?? 'Loading…'}</span>
                  </div>
                  <p className="font-prose mt-0.5 text-xs text-muted">
                    {mobile
                      ? `Differential drive · ${robot?.wheels.length ?? 0} wheels · lidar, IMU and camera in Gazebo`
                      : `Arm · ${movable.length} movable joints`}
                  </p>
                  <button
                    onClick={() => {
                      setPanel('robot');
                      setDraft('raise the arm 45 degrees');
                    }}
                    className="mt-2 w-full border border-line bg-surfaceAlt px-2.5 py-1.5 text-left text-xs transition hover:border-brand focus-visible:outline focus-visible:outline-1 focus-visible:outline-brand"
                  >
                    <span className="text-brand">&rsaquo;</span> It listens &mdash; try{' '}
                    <span className="text-brand">&quot;raise the arm 45 degrees&quot;</span> in the console
                  </button>
                </section>

                {/* 2. Kinematic chain - open by default; the project is about this. */}
                {robot && (
                  <details open>
                    <summary className="cursor-pointer list-none border-t border-line pt-3 text-xs font-bold uppercase tracking-widest text-brand focus-visible:outline focus-visible:outline-1 focus-visible:outline-brand">
                      Kinematic chain &middot; {movable.filter((j) => !(robot.wheels ?? []).includes(j.name)).length} joints
                    </summary>
                    <div className="mt-2 space-y-2">
                      {movable
                        .filter((j) => !(robot.wheels ?? []).includes(j.name))
                        .map((j) => {
                          const v = snapshot.joints[j.name] ?? 0;
                          const bounded =
                            j.type !== 'continuous' &&
                            j.lower_limit != null &&
                            j.upper_limit != null &&
                            j.upper_limit > j.lower_limit;
                          const pct = bounded
                            ? ((v - j.lower_limit!) / (j.upper_limit! - j.lower_limit!)) * 100
                            : 50;
                          return (
                            <div key={j.name} className="text-xs">
                              <div className="flex items-baseline justify-between gap-2">
                                <span className="truncate font-medium">{j.name}</span>
                                <span className="shrink-0 tabular-nums">
                                  {v.toFixed(2)}
                                  <span className="ml-0.5 text-muted">{j.type === 'prismatic' ? 'm' : 'rad'}</span>
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="w-8 shrink-0 text-[10px] text-muted">{j.type.slice(0, 4)}</span>
                                <div className="h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-line">
                                  <div className="h-full bg-brand" style={{ width: `${Math.min(100, Math.max(2, pct))}%` }} />
                                </div>
                                <span className="shrink-0 tabular-nums text-[10px] text-muted">
                                  {bounded ? `${j.lower_limit!.toFixed(2)}…${j.upper_limit!.toFixed(2)}` : 'free'}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      {!movable.length && <p className="text-xs text-muted">No joints yet.</p>}
                    </div>
                  </details>
                )}

                {/* 3. Sensors - the number itself shows proximity. */}
                {mobile && (
                  <details open>
                    <summary className="cursor-pointer list-none border-t border-line pt-3 text-xs font-bold uppercase tracking-widest text-brand focus-visible:outline focus-visible:outline-1 focus-visible:outline-brand">
                      Sensors
                    </summary>
                    <div className="mt-2 flex items-baseline justify-between text-sm">
                      <span className="flex items-center gap-2 text-xs text-muted">
                        <Radar size={14} className="text-brand" /> Distance ahead
                      </span>
                      <span
                        className={`text-lg font-bold tabular-nums transition-colors ${
                          front === undefined || front >= 5.99
                            ? 'text-fg'
                            : front < 0.6
                              ? 'text-danger'
                              : front < 1.5
                                ? 'text-warn'
                                : 'text-brand'
                        }`}
                      >
                        {front === undefined ? '–' : front >= 5.99 ? 'clear' : `${front.toFixed(2)} m`}
                      </span>
                    </div>
                    <p className="font-prose mt-1 text-xs text-muted">
                      Commands like &quot;drive until you see the wall&quot; read this sensor.
                    </p>
                  </details>
                )}

                {/* 4. Scene - what is placed, click to select. */}
                <details open={objects.length > 0}>
                  <summary className="cursor-pointer list-none border-t border-line pt-3 text-xs font-bold uppercase tracking-widest text-muted focus-visible:outline focus-visible:outline-1 focus-visible:outline-brand">
                    Scene &middot; {objects.length}
                  </summary>
                  <div className="mt-2 space-y-1">
                    {objects.map((o) => (
                      <button
                        key={o.id}
                        onClick={() => selectObject(o.id)}
                        className={`flex w-full min-w-0 items-center gap-2 px-1 py-0.5 text-left text-xs transition hover:text-brand focus-visible:outline focus-visible:outline-1 focus-visible:outline-brand ${
                          selectedId === o.id ? 'text-brand' : 'text-fg'
                        }`}
                      >
                        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: o.color }} />
                        <span className="truncate">{o.name}</span>
                        <span className="ml-auto shrink-0 text-[10px] text-muted">{o.kind}</span>
                      </button>
                    ))}
                    {!objects.length && (
                      <p className="font-prose text-xs text-muted">Nothing placed yet &mdash; use the Shapes tab.</p>
                    )}
                  </div>
                </details>

                {/* 5. Load robot - demoted to the bottom. */}
                <section>
                  <h2 className="mb-2 border-t border-line pt-3 text-xs font-bold uppercase tracking-widest text-muted">
                    Load robot
                  </h2>
                  <input
                    ref={fileInput}
                    type="file"
                    multiple
                    accept={UPLOAD_ACCEPT}
                    className="hidden"
                    onChange={(e) => {
                      void upload(Array.from(e.target.files ?? []));
                      e.target.value = '';
                    }}
                  />
                  <input
                    ref={folderInput}
                    type="file"
                    className="hidden"
                    // @ts-expect-error webkitdirectory is non-standard but widely supported
                    webkitdirectory=""
                    onChange={(e) => {
                      void upload(Array.from(e.target.files ?? []));
                      e.target.value = '';
                    }}
                  />
                  <button
                    onClick={() => fileInput.current?.click()}
                    disabled={busy || !health}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-line px-4 py-2 text-sm font-semibold text-fg transition hover:border-brand hover:text-brand disabled:opacity-50"
                  >
                    {busy ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                    Upload URDF / Xacro / ZIP
                  </button>
                  <button
                    onClick={() => folderInput.current?.click()}
                    disabled={busy || !health}
                    className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl border border-line px-4 py-2 text-sm font-semibold text-muted transition hover:border-brand hover:text-brand disabled:opacity-50"
                  >
                    <FolderUp size={15} />
                    Upload a robot folder
                  </button>
                  <p className="font-prose mt-1.5 text-xs text-muted">.urdf, .xacro or .zip &mdash; or drop files onto the viewport.</p>
                </section>
              </div>
            ) : (
              <ShapesPanel
                objects={objects}
                selectedId={selectedId}
                onAdd={addObject}
                onChange={changeObject}
                onDelete={deleteObject}
                onSelect={selectObject}
                scale={robot?.scale}
              />
            )}
          </div>
        </aside>

        {/* Center: viewport */}
        <main
          className="relative order-1 h-[46vh] min-h-0 min-w-0 overflow-hidden lg:order-2 lg:h-auto"
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <SceneViewport
            info={robot}
            latest={latest}
            objects={objects}
            selectedId={selectedId}
            onSelect={selectObject}
            onMove={moveObject}
            traceKey={traceKey}
            plannedPath={plannedPath}
            tool={tool}
            cameraScale={Math.max(1, robot?.scale?.factor ?? 1)}
            follow={follow}
            viewKey={viewKey}
            onTransform={transformObject}
            onSnapshot={(dataUrl) => update(id, { thumb: dataUrl })}
          />
          <ViewportToolbar
            tool={tool}
            onTool={setTool}
            follow={follow}
            onFollow={() => setFollow((f) => !f)}
            onFit={() => setViewKey((n) => n + 1)}
            canDelete={!!selectedId}
            hasSelection={!!selectedId}
            onDelete={() => selectedId && deleteObject(selectedId)}
          />
          {dragging && (
            <div className="pointer-events-none absolute inset-3 flex items-center justify-center rounded-2xl border-4 border-dashed border-brand bg-brand/15 text-lg font-bold text-brand">
              Drop URDF / Xacro / ZIP to load
            </div>
          )}
        </main>

        {/* Right: commands */}
        <aside className="order-3 flex min-h-0 flex-col border-line bg-surface lg:border-l">
          <div className="grid shrink-0 grid-cols-3 border-b border-line">
            {(
              [
                ['robot', 'Tell the robot', Wand2],
                ['map', 'Build a map', MapIcon],
                ['console', 'Console', Terminal],
              ] as const
            ).map(([key, label, Icon]) => (
              <button
                key={key}
                onClick={() => {
                  setPanel(key);
                  setDraft('');
                }}
                className={`flex items-center justify-center gap-2 border-b-2 px-3 py-3 text-xs font-bold uppercase tracking-wider transition ${
                  panel === key ? 'border-brand text-brand' : 'border-transparent text-muted hover:text-fg'
                }`}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {panel === 'console' && (
            <div className="min-h-0 flex-1 overflow-hidden p-3">
              <ConsolePanel snapshot={snapshot} robot={robot} />
            </div>
          )}

          <div className={panel === 'console' ? 'hidden' : 'min-h-0 flex-1 space-y-3 overflow-y-auto p-4'}>
            {panel === 'robot' && activePlan.length > 0 && (
              <div className="sticky top-0 z-10 -mx-1 px-1 pb-1" style={{ background: 'rgb(var(--surface))' }}>
                <PlanSteps steps={activePlan} snapshot={snapshot} />
              </div>
            )}
            {!list.length && (
              <div className="space-y-2">
                <p className="text-sm text-muted">
                  {panel === 'map' ? 'Describe a map and I will build it out of shapes:' : 'Type what the robot should do - it runs on real ROS 2 topics. Or start with:'}
                </p>
                {panel === 'robot' && (
                  <button
                    onClick={runDemo}
                    disabled={busy || !health}
                    className="flex w-full items-center gap-2 border border-brand bg-brand/10 px-3 py-2.5 text-left text-sm font-bold text-brand transition hover:bg-brand/20 disabled:opacity-50"
                  >
                    <span>&#9654;</span> Run the demo - wall + drive + turn, one click
                  </button>
                )}
                {(panel === 'map' ? MAP_SUGGESTIONS : suggestions).map((s) => (
                  <button
                    key={s}
                    onClick={() => void send(s)}
                    disabled={!health}
                    className="block w-full rounded-xl border border-line bg-surfaceAlt px-3 py-2 text-left text-sm transition hover:border-brand hover:bg-brand/10 disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
                {panel === 'robot' && mobile && !objects.length && (
                  <p className="pt-1 text-xs text-muted">
                    Tip: open the <b>Shapes</b> tab and add a Wall in front of the rover first.
                  </p>
                )}
              </div>
            )}

            {list.map((m) => (
              <div key={m.id} className="space-y-1.5">
                <div className="ml-8 rounded-2xl rounded-br-sm bg-brand px-3 py-2 text-sm text-white shadow">{m.text}</div>
                <div className="mr-8 rounded-2xl rounded-bl-sm border border-line bg-surfaceAlt px-3 py-2 text-sm">
                  {m.status === 'pending' && (
                    <span className="flex items-center gap-2 text-muted">
                      <Loader2 size={14} className="animate-spin" />{' '}
                      {panel === 'map' ? 'Building the map… big ones can take up to a minute' : 'Planning…'}
                    </span>
                  )}
                  {m.status === 'error' && <span className="text-danger">{m.error}</span>}
                  {m.status === 'ok' && m.result && (
                    <div>
                      {m.result.reply && <div>{m.result.reply}</div>}
                      <div className="mt-1 text-xs font-semibold text-accent">
                        {panel === 'map' ? (
                          `${m.result.world?.length ?? 0} objects in the map`
                        ) : (
                          <>
                            {m.result.plan.length} step{m.result.plan.length === 1 ? '' : 's'}
                            {m.result.repeat ? ' (repeating)' : ''}
                            {m.result.world ? ' · world updated' : ''}
                          </>
                        )}
                      </div>
                      <div className="text-xs text-muted">
                        {m.result.llm.provider
                          ? `AI: ${m.result.llm.provider}`
                          : 'Basic parser (no AI connected: add ANTHROPIC_API_KEY in backend/.env)'}
                        {m.result.llm_error ? ` · ${m.result.llm_error}` : ''}
                      </div>
                      {m.result.warnings.map((w) => (
                        <div key={w} className="text-xs text-warn">{w}</div>
                      ))}
                      <details className="mt-1" hidden={panel === 'map'}>
                        <summary className="cursor-pointer text-xs text-brand">Show plan</summary>
                        <pre className="mt-1 max-h-48 overflow-auto rounded-lg bg-surface p-2 text-[11px] leading-snug">
                          {JSON.stringify(m.result.plan, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={logEnd} />
          </div>

          <form onSubmit={onSubmit} className={panel === 'console' ? 'hidden' : 'border-t border-line p-3'}>
            {panel === 'map' && objects.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setObjects([]);
                  setSelectedId(null);
                }}
                className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-danger hover:underline"
              >
                <Trash2 size={12} /> Clear the map ({objects.length} objects)
              </button>
            )}
            {panel === 'robot' && snapshot.queued > 0 && <div className="mb-2 text-xs text-muted">{snapshot.queued} plan(s) queued</div>}
            <div className="flex items-end gap-2 rounded-xl border border-line bg-surfaceAlt p-2 focus-within:border-brand">
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void send(draft);
                  }
                }}
                rows={2}
                placeholder={
                  panel === 'map'
                    ? 'e.g. build a maze with a few dead ends'
                    : mobile
                      ? 'e.g. drive forward until you see the wall, then turn left'
                      : 'e.g. flap for 3 seconds, then wait 2 seconds'
                }
                className="flex-1 resize-none bg-transparent text-sm text-fg outline-none placeholder:text-muted"
              />
              <button
                type="submit"
                disabled={!draft.trim() || !health}
                aria-label={panel === 'map' ? 'Build map' : 'Send command'}
                className="rounded-lg bg-brand p-2.5 text-white shadow transition hover:brightness-110 disabled:opacity-40"
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        </aside>
      </div>
    </div>
  );
}
