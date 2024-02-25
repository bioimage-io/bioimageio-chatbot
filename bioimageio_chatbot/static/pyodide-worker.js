const indexURL = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/'
importScripts(`${indexURL}pyodide.js`);

(async () => {
    self.pyodide = await loadPyodide({ indexURL })
    await self.pyodide.loadPackage("micropip");
    const micropip = self.pyodide.pyimport("micropip");
    await micropip.install(['numpy', 'imjoy-rpc', 'pyodide-http']);
    // NOTE: We intentionally avoid runPythonAsync here because we don't want this to pre-load extra modules like matplotlib.
    self.pyodide.runPython(setupCode)
    self.postMessage({loading: true})  // Inform the main thread that we finished loading.
})()

let outputs = []

function write(type, content) {
    self.postMessage({ type, content })
    outputs.push({ type, content })
    return content.length
}

function logService(type, url, attrs) {
    outputs.push({type, content: url, attrs: attrs?.toJs({dict_converter : Object.fromEntries})})
    self.postMessage({ type, content: url, attrs: attrs?.toJs({dict_converter : Object.fromEntries}) })
}

function show(type, url, attrs) {
    const turl = url.length > 32 ? url.slice(0, 32) + "..." : url
    outputs.push({type, content: turl, attrs: attrs?.toJs({dict_converter : Object.fromEntries})})
    self.postMessage({ type, content: url, attrs: attrs?.toJs({dict_converter : Object.fromEntries}) })
}

// Stand-in for `time.sleep`, which does not actually sleep.
// To avoid a busy loop, instead import asyncio and await asyncio.sleep().
function spin(seconds) {
    const time = performance.now() + seconds * 1000
    while (performance.now() < time);
}

// NOTE: eval(compile(source, "<string>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
// returns a coroutine if `source` contains a top-level await, and None otherwise.

const setupCode = `
import array
import ast
import base64
import contextlib
import io
import js
import pyodide
import sys
import time
import traceback
import wave
import pyodide_http

pyodide_http.patch_all()  # Patch all libraries

time.sleep = js.spin

# patch hypha services
import imjoy_rpc.hypha
_connect_to_server = imjoy_rpc.hypha.connect_to_server

async def patched_connect_to_server(*args, **kwargs):
    server = await _connect_to_server(*args, **kwargs)
    _register_service = server.register_service
    async def patched_register_service(*args, **kwargs):
        svc_info = await _register_service(*args, **kwargs)
        service_id = svc_info['id'].split(':')[1]
        service_url = f"{server.config['public_base_url']}/{server.config['workspace']}/services/{service_id}"
        js.logService("service", service_url, svc_info)
        return svc_info
    server.register_service = patched_register_service
    server.registerService = patched_register_service
    return server

imjoy_rpc.hypha.connect_to_server = patched_connect_to_server

# For redirecting stdout and stderr later.
class JSOutWriter(io.TextIOBase):
    def write(self, s):
        return js.write("stdout", s)

class JSErrWriter(io.TextIOBase):
    def write(self, s):
        return js.write("stderr", s)

def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def show():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
        js.show("img", img)
        plt.clf()

    plt.show = show

def show_image(image, **attrs):
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format='png')
    data = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
    js.show("img", data, attrs)

def show_animation(frames, duration=100, format="apng", loop=0, **attrs):
    from PIL import Image
    buf = io.BytesIO()
    img, *imgs = [frame if isinstance(frame, Image.Image) else Image.fromarray(frame) for frame in frames]
    img.save(buf, format='png' if format == "apng" else format, save_all=True, append_images=imgs, duration=duration, loop=0)
    img = f'data:image/{format};base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
    js.show("img", img, attrs)

def convert_audio(data):
    try:
        import numpy as np
        is_numpy = isinstance(data, np.ndarray)
    except ImportError:
        is_numpy = False
    if is_numpy:
        if len(data.shape) == 1:
            channels = 1
        if len(data.shape) == 2:
            channels = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError("Too many dimensions (expected 1 or 2).")
        return ((data * (2**15 - 1)).astype("<h").tobytes(), channels)
    else:
        data = array.array('h', (int(x * (2**15 - 1)) for x in data))
        if sys.byteorder == 'big':
            data.byteswap()
        return (data.tobytes(), 1)

def show_audio(samples, rate):
    bytes, channels = convert_audio(samples)
    buf = io.BytesIO()
    with wave.open(buf, mode='wb') as w:
        w.setnchannels(channels)
        w.setframerate(rate)
        w.setsampwidth(2)
        w.setcomptype('NONE', 'NONE')
        w.writeframes(bytes)
    audio = 'data:audio/wav;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
    js.show("audio", audio)

# HACK: Prevent 'wave' import from failing because audioop is not included with pyodide.
import types
embed = types.ModuleType('embed')
sys.modules['embed'] = embed
embed.image = show_image
embed.animation = show_animation
embed.audio = show_audio
context = {}  # Persistent execution context

async def run(source):
    out = JSOutWriter()
    err = JSErrWriter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            imports = pyodide.code.find_imports(source)
            await js.pyodide.loadPackagesFromImports(source)
            if "matplotlib" in imports or "skimage" in imports:
                setup_matplotlib()
            if "embed" in imports:
                await js.pyodide.loadPackagesFromImports("import numpy, PIL")
            code = compile(source, "<string>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            result = eval(code, context)
            if result:
                await result
        except:
            traceback.print_exc()
`

self.onmessage = async (event) => {
    const mountedFs = {}
    if(event.data.source){
        try{
            const { source } = event.data
            self.pyodide.globals.set("source", source)
            outputs = []
            await self.pyodide.runPythonAsync("await run(source)")
            self.postMessage({ executionDone: true, outputs })
            // synchronize the file system
            for(const mountPoint of Object.keys(mountedFs)){
                await mountedFs[mountPoint].syncfs()
            }
            outputs = []
        }
        catch(e){
            self.postMessage({ executionError: e.message })
        }
    }
    if(event.data.mount){
        try{
            const { mountPoint, dirHandle } = event.data.mount
            const nativefs = await self.pyodide.mountNativeFS(mountPoint, dirHandle)
            mountedFs[mountPoint] = nativefs
            console.log("Native FS mounted:", nativefs)
            self.postMessage({ mounted: mountPoint })
        }
        catch(e){
            self.postMessage({ mountError: e.message })
        }
    }

}
