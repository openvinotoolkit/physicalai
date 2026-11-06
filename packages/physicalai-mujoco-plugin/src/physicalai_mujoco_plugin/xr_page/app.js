// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The headset's teleoperation page (xr.py): draws the simulation in 3D from the geometry and poses
// the server streams, shows its cameras, and sends the controllers and where it shows the robot.
"use strict";

const statusLine = document.getElementById("status");
const start = document.getElementById("start");
const preview = document.getElementById("preview");
const canvas = document.getElementById("view");
const STICK_DEAD = 0.15;
const GRIP_HELD = 0.4; // a grip pressed past this holds an arm: the robot stays put meanwhile (teleop.GRIP_OFF)
const RAISE_SPEED = 0.3; // m/s at full thumbstick
const PUSH_SPEED = 0.4;
let socket = null;
let mode = null;
let cameras = [];
const frames = {};
let scene = null; // the newest geometry and view, as fetched
let poses = null; // the newest poses: version, then x y z qw qx qy qz per body
let sceneLoading = false;
let sceneFailed = -Infinity;
let camerasChanged = false; // the camera list changed: the headset lays the panels out again
let sceneSerial = 0; // counts the scenes loaded: a reconnected server may reuse a version number
let epoch = 0; // counts the connections: a scene asked for before a reconnect is dropped
let session = null;
let code = new URLSearchParams(location.search).get("code") || localStorage.getItem("xrPairingCode") || "";
const pairing = document.getElementById("pairing");
const codeInput = document.getElementById("code");

function withCode(path) {
  return `${path}${path.includes("?") ? "&" : "?"}code=${encodeURIComponent(code)}`;
}

// Pairing: the code the simulation logs, asked for once and kept in this browser.
async function pair() {
  let paired = false;
  if (code) {
    try {
      paired = (await fetch(withCode("api/check"), { cache: "no-store" })).ok;
    } catch (error) {
      paired = false;
    }
  }
  if (!paired) {
    pairing.hidden = false;
    statusLine.textContent = code
      ? "That pairing code is wrong: enter the one in the simulation's log or the viewer's Teleop folder."
      : "Enter the pairing code from the simulation's log or the viewer's Teleop folder.";
    return;
  }
  localStorage.setItem("xrPairingCode", code);
  pairing.hidden = true;
  connect(); // each connection lists the cameras
}
pairing.onsubmit = (event) => {
  event.preventDefault();
  code = codeInput.value.trim();
  pair();
};

// --- The simulation: the WebSocket, the geometry, the camera frames ---------------------------------
function connect() {
  socket = new WebSocket(withCode(`wss://${location.host}/ws`));
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    statusLine.textContent = "Connected to the simulation.";
    // The simulation may have restarted with another robot: forget its geometry and poses.
    epoch++;
    scene = null;
    poses = null;
    loadCameras();
    sceneLoading = false;
  };
  socket.onclose = () => {
    statusLine.textContent = "Disconnected; retrying…";
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    if (!(event.data instanceof ArrayBuffer)) return;
    poses = new Float32Array(event.data);
    if (!scene || scene.version !== poses[0]) loadScene();
  };
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 8192) {
    socket.send(JSON.stringify(message));
  }
}

async function loadScene() {
  if (sceneLoading || performance.now() - sceneFailed < 2000) return; // one request at a time; wait after a failure
  sceneLoading = true;
  const asked = epoch;
  try {
    const response = await fetch(withCode("api/scene"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const length = new Uint32Array(buffer, 0, 1)[0];
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
    const first = 4 + length;
    if (asked !== epoch) return; // asked before a reconnect
    scene = {
      ...header,
      serial: ++sceneSerial,
      epoch: asked,
      vertexData: new Uint8Array(buffer, first, header.vertices * 16),
      indexData: new Uint32Array(buffer, first + header.vertices * 16, header.indices),
    };
  } catch (error) {
    console.warn("Could not load the scene", error);
    sceneFailed = performance.now();
  } finally {
    if (asked === epoch) sceneLoading = false;
  }
}

// Each camera's newest JPEG, fetched one after the other, about 20 per second, for as long as the
// connection that listed it lasts.
async function fetchCamera(name, listed) {
  while (listed === epoch) {
    try {
      const response = await fetch(withCode(`api/camera/${encodeURIComponent(name)}.jpg`), { cache: "no-store" });
      if (response.ok && listed === epoch) {
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);
        if (frames[name] && frames[name].bitmap) frames[name].bitmap.close();
        frames[name] = { bitmap, fresh: true };
        if (name === cameras[0]) {
          if (preview.src) URL.revokeObjectURL(preview.src);
          preview.src = URL.createObjectURL(blob);
        }
      }
    } catch (error) {
      // The simulation restarts or switches scenes: try again.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

// The cameras of the simulation behind this connection: a restarted one may have others.
async function loadCameras() {
  const listed = epoch;
  let names = [];
  try {
    names = await (await fetch(withCode("api/cameras"))).json();
  } catch (error) {
    names = [];
  }
  if (listed !== epoch) return;
  for (const name of Object.keys(frames)) {
    if (frames[name].bitmap) frames[name].bitmap.close();
    delete frames[name];
  }
  for (const name of Object.keys(textures)) {
    if (gl) gl.deleteTexture(textures[name]);
    delete textures[name];
  }
  cameras = names;
  camerasChanged = true;
  for (const name of cameras) fetchCamera(name, listed);
}

// --- Matrices (column-major, as WebGL takes them) ---------------------------------------------------
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

function bodyMatrix(p, body) {
  const o = 1 + 7 * body;
  const w = p[o + 3], x = p[o + 4], y = p[o + 5], z = p[o + 6];
  return new Float32Array([
    1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0,
    2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0,
    2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0,
    p[o], p[o + 1], p[o + 2], 1,
  ]);
}

// Where the robot stands in the room. The operator's eye point (scene.view.eye) lands DISTANCE ahead
// of the head and HEIGHT above it, and the robot's heading (scene.view.yaw) points along the level
// direction (ax, az) the operator looked at the start.
function placeFrom(viewer, previous) {
  const p = viewer.transform.position, o = viewer.transform.orientation;
  let ax = -2 * (o.x * o.z + o.w * o.y), az = -(1 - 2 * (o.x * o.x + o.y * o.y));
  const length = Math.hypot(ax, az) || 1;
  ax /= length;
  az /= length;
  return {
    ax, az, hx: p.x, hy: p.y, hz: p.z,
    distance: previous ? previous.distance : 0,
    height: previous ? previous.height : 0,
  };
}

function anchor(place) {
  return [place.hx + place.ax * place.distance, place.hy + place.height, place.hz + place.az * place.distance];
}

// The robot's world frame in the room (xr.xr_to_robot, inverted): robot ahead (x) along (ax, az),
// left (y) to the operator's left, up (z) up, turned by -yaw and moved so that the eye point lands
// at the anchor.
function placement(place, view) {
  const c = Math.cos(view.yaw), s = Math.sin(view.yaw);
  const { ax, az } = place;
  const x = [c * ax - s * az, 0, c * az + s * ax];
  const y = [s * ax + c * az, 0, s * az - c * ax];
  const z = [0, 1, 0];
  const [px, py, pz] = anchor(place);
  const e = view.eye;
  const t = [0, 1, 2].map((i) => [px, py, pz][i] - (x[i] * e[0] + y[i] * e[1] + z[i] * e[2]));
  return new Float32Array([...x, 0, ...y, 0, ...z, 0, ...t, 1]);
}

function yaw(place) {
  const half = Math.atan2(-place.ax, -place.az) / 2;
  return [Math.cos(half), 0, Math.sin(half), 0];
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), d = near - far;
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / d, -1, 0, 0, (2 * far * near) / d, 0]);
}

function lookAt(eye, target) {
  let fx = eye[0] - target[0], fy = eye[1] - target[1], fz = eye[2] - target[2]; // back
  let n = Math.hypot(fx, fy, fz);
  fx /= n; fy /= n; fz /= n;
  let rx = fz, rz = -fx; // right: up x back
  n = Math.hypot(rx, rz);
  rx /= n; rz /= n;
  const ux = fy * rz, uy = fz * rx - fx * rz, uz = -fy * rx; // up: back x right
  const dot = (a, b, c) => -(a * eye[0] + b * eye[1] + c * eye[2]);
  return new Float32Array([
    rx, ux, fx, 0, 0, uy, fy, 0, rz, uz, fz, 0, dot(rx, 0, rz), dot(ux, uy, uz), dot(fx, fy, fz), 1,
  ]);
}

// --- Drawing ----------------------------------------------------------------------------------------
const gl = canvas.getContext("webgl2", { xrCompatible: true, antialias: true });

function program(vertex, fragment, attributes) {
  const make = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const p = gl.createProgram();
  gl.attachShader(p, make(gl.VERTEX_SHADER, vertex));
  gl.attachShader(p, make(gl.FRAGMENT_SHADER, fragment));
  attributes.forEach((name, index) => gl.bindAttribLocation(p, index, name));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));
  return p;
}

const MESH_VERTEX = `#version 300 es
in vec3 position; in vec3 normal;
uniform mat4 viewProjection; uniform mat4 model;
out vec3 worldNormal;
void main() {
  worldNormal = mat3(model) * normal;
  gl_Position = viewProjection * model * vec4(position, 1.0);
}`;
const MESH_FRAGMENT = `#version 300 es
precision mediump float;
in vec3 worldNormal; uniform vec4 color; out vec4 outColor;
const vec3 light = vec3(0.267, 0.802, 0.535);
void main() {
  float shade = 0.42 + 0.58 * abs(dot(normalize(worldNormal), light));
  outColor = vec4(color.rgb * shade * color.a, color.a);
}`;
const PANEL_VERTEX = `attribute vec2 corner; uniform mat4 transform; varying vec2 uv;
void main() {
  uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
  gl_Position = transform * vec4(corner, 0.0, 1.0);
}`;
const PANEL_FRAGMENT = `precision mediump float; varying vec2 uv; uniform sampler2D image;
void main() { gl_FragColor = texture2D(image, uv); }`;

let meshes = null;
let uniforms = null;
let panelShader = null;
let panelTransform = null;
let quad = null;
const textures = {};
if (gl) {
  meshes = program(MESH_VERTEX, MESH_FRAGMENT, ["position", "normal"]);
  uniforms = {
    viewProjection: gl.getUniformLocation(meshes, "viewProjection"),
    model: gl.getUniformLocation(meshes, "model"),
    color: gl.getUniformLocation(meshes, "color"),
  };
  panelShader = program(PANEL_VERTEX, PANEL_FRAGMENT, ["corner"]);
  panelTransform = gl.getUniformLocation(panelShader, "transform");
  quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
}

let uploaded = null;
function upload() {
  if (!scene || (uploaded && uploaded.serial === scene.serial)) return;
  if (uploaded) {
    gl.deleteVertexArray(uploaded.vao);
    gl.deleteBuffer(uploaded.vertices);
    gl.deleteBuffer(uploaded.indices);
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vertices = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
  gl.bufferData(gl.ARRAY_BUFFER, scene.vertexData, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.BYTE, true, 16, 12);
  const indices = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, scene.indexData, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  uploaded = { serial: scene.serial, epoch: scene.epoch, version: scene.version, vao, vertices, indices, geoms: scene.geoms, view: scene.view };
}

// Each body's matrix in the room: the placement times the body's pose; null until both have arrived.
function bodyMatrices(place) {
  upload();
  // Only this connection's geometry: a restarted simulation may send the same version for another robot.
  if (!uploaded || uploaded.epoch !== epoch || !poses || poses[0] !== uploaded.version) return null;
  const room = placement(place, uploaded.view);
  const out = [];
  for (let body = 0; body < (poses.length - 1) / 7; body++) out.push(multiply(room, bodyMatrix(poses, body)));
  return out;
}

function drawMeshes(viewProjection, matrices, seeThrough) {
  if (!matrices) return;
  gl.useProgram(meshes);
  gl.uniformMatrix4fv(uniforms.viewProjection, false, viewProjection);
  gl.bindVertexArray(uploaded.vao);
  if (seeThrough) {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // the colours are premultiplied by alpha
    gl.depthMask(false);
  }
  for (const [body, r, g, b, a, first, count] of uploaded.geoms) {
    if (a < 1 !== seeThrough) continue;
    gl.uniformMatrix4fv(uniforms.model, false, matrices[body]);
    gl.uniform4f(uniforms.color, r, g, b, a);
    gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, first * 4);
  }
  gl.bindVertexArray(null);
  gl.depthMask(true);
  gl.disable(gl.BLEND);
}

// Camera panels: the first camera in the middle, the others on its sides, above the robot.
function layout(place) {
  const rx = -place.az, rz = place.ax; // right of ahead
  const panels = [];
  const put = (name, along, side, height, width) => {
    const x = place.hx + place.ax * along + rx * side, z = place.hz + place.az * along + rz * side;
    panels.push({ name, x, y: place.hy + height, z, rx, rz, ax: place.ax, az: place.az, width });
  };
  if (cameras.length) put(cameras[0], 1.4, 0, 0.3, 0.9);
  const rest = cameras.slice(1);
  rest.forEach((name, index) => put(name, 1.3, (index - (rest.length - 1) / 2) * 1.5, 0.3, 0.5));
  return panels;
}

function panelMatrix(panel, aspect) {
  const w = panel.width / 2, h = w / aspect;
  return new Float32Array([
    panel.rx * w, 0, panel.rz * w, 0, 0, h, 0, 0, -panel.ax, 0, -panel.az, 0, panel.x, panel.y, panel.z, 1,
  ]);
}

function updateTextures() {
  for (const name of cameras) {
    const entry = frames[name];
    if (!entry || !entry.fresh) continue;
    if (!textures[name]) {
      textures[name] = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, textures[name]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, textures[name]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, entry.bitmap);
    textures[name].aspect = entry.bitmap.width / entry.bitmap.height;
    entry.fresh = false;
  }
}

function drawPanels(viewProjection, panels) {
  gl.useProgram(panelShader);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  for (const panel of panels) {
    const texture = textures[panel.name];
    if (!texture) continue;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniformMatrix4fv(panelTransform, false, multiply(viewProjection, panelMatrix(panel, texture.aspect)));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

function drawView(viewProjection, matrices, panels) {
  drawMeshes(viewProjection, matrices, false);
  if (panels) drawPanels(viewProjection, panels);
  drawMeshes(viewProjection, matrices, true);
}

// --- On this page: what the headset shows at the start, from where the operator stands -------------
const PAGE_PLACE = { ax: 0, az: -1, hx: 0, hy: 1.6, hz: 0, distance: 0, height: 0 };
function drawPage() {
  if (!session) {
    const scale = window.devicePixelRatio || 1;
    const width = Math.round(canvas.clientWidth * scale), height = Math.round(canvas.clientHeight * scale);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.08, 0.09, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const eye = [0, 1.6, 0];
    const target = [0, 1.25, -0.4]; // ahead and down, where the arms reach
    const viewProjection = multiply(perspective(1.0, width / Math.max(height, 1), 0.05, 20), lookAt(eye, target));
    drawView(viewProjection, bodyMatrices(PAGE_PLACE), null);
  }
  requestAnimationFrame(drawPage);
}

// --- In the headset ---------------------------------------------------------------------------------
function pose(p) {
  if (!p) return null;
  const t = p.transform;
  const o = t.orientation;
  return { p: [t.position.x, t.position.y, t.position.z], q: [o.w, o.x, o.y, o.z] };
}

async function detect() {
  if (!gl) {
    statusLine.textContent = "This browser has no WebGL 2.";
    return;
  }
  requestAnimationFrame(drawPage);
  if (!navigator.xr) {
    statusLine.textContent = "This browser has no WebXR: open the page in the headset's browser.";
    return;
  }
  if (await navigator.xr.isSessionSupported("immersive-ar")) mode = "immersive-ar";
  else if (await navigator.xr.isSessionSupported("immersive-vr")) mode = "immersive-vr";
  start.disabled = mode === null;
}

// Reads the controllers: their poses, triggers and grips for the server, and the buttons and
// thumbsticks that move the robot and toggle the cameras.
function readInputs(xrFrame, space) {
  const inputs = { controllers: {}, recenter: false, panels: false, raise: 0, push: 0, holding: false };
  for (const source of xrFrame.session.inputSources) {
    const buttons = source.gamepad ? source.gamepad.buttons : [];
    const axes = source.gamepad ? source.gamepad.axes : [];
    const value = (index) => (buttons[index] ? buttons[index].value : 0);
    if (buttons[4] && buttons[4].pressed) inputs.recenter = true;
    if (buttons[5] && buttons[5].pressed) inputs.panels = true;
    if (value(1) > GRIP_HELD) inputs.holding = true;
    const stick = axes.length > 3 && Math.abs(axes[3]) > STICK_DEAD ? -axes[3] : 0;
    if (source.handedness === "left") inputs.raise = stick;
    if (source.handedness === "right") inputs.push = stick;
    if (!source.gripSpace || (source.handedness !== "left" && source.handedness !== "right")) continue;
    const grip = pose(xrFrame.getPose(source.gripSpace, space));
    if (grip) inputs.controllers[source.handedness] = { ...grip, trigger: value(0), squeeze: value(1) };
  }
  return inputs;
}

start.onclick = async () => {
  session = await navigator.xr.requestSession(mode, { optionalFeatures: ["local-floor"] });
  await session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  let space;
  try {
    space = await session.requestReferenceSpace("local-floor");
  } catch (error) {
    space = await session.requestReferenceSpace("local");
  }
  let place = null;
  let panels = null;
  let showPanels = true;
  const held = { recenter: false, panels: false };
  let last = null;
  session.addEventListener("end", () => {
    send({ type: "end" });
    session = null;
  });
  session.requestAnimationFrame(function frame(time, xrFrame) {
    xrFrame.session.requestAnimationFrame(frame);
    const dt = last === null ? 0 : Math.min((time - last) / 1000, 0.1);
    last = time;
    const viewer = xrFrame.getViewerPose(space);
    const inputs = readInputs(xrFrame, space);
    // The robot stays put while the operator holds an arm: moving it would move the arm's target.
    if (viewer && (place === null || (inputs.recenter && !held.recenter && !inputs.holding))) {
      place = placeFrom(viewer, place);
      panels = null;
    }
    if (inputs.panels && !held.panels) showPanels = !showPanels;
    held.recenter = inputs.recenter;
    held.panels = inputs.panels;
    if (place && !inputs.holding) {
      place.height = Math.min(Math.max(place.height + inputs.raise * RAISE_SPEED * dt, -2), 1);
      place.distance = Math.min(Math.max(place.distance + inputs.push * PUSH_SPEED * dt, -2), 3);
    }
    if (camerasChanged) {
      panels = null;
      camerasChanged = false;
    }
    if (place && panels === null && cameras.length) panels = layout(place);
    const message = { type: "frame", head: pose(viewer), ...inputs.controllers };
    if (place) message.frame = { p: anchor(place), q: yaw(place) };
    send(message);

    const layer = xrFrame.session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    if (mode === "immersive-ar") gl.clearColor(0, 0, 0, 0);
    else gl.clearColor(0.08, 0.09, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!viewer || !place) return;
    gl.enable(gl.DEPTH_TEST);
    updateTextures();
    const matrices = bodyMatrices(place);
    for (const view of viewer.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      const viewProjection = multiply(view.projectionMatrix, view.transform.inverse.matrix);
      drawView(viewProjection, matrices, showPanels ? panels : null);
    }
  });
};

detect();
pair();
