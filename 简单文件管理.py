from flask import Flask, request, jsonify, send_from_directory, render_template_string, abort  # Flask modules
import os  # OS file operations
from flask_httpauth import HTTPBasicAuth  # Added for auth
from werkzeug.security import generate_password_hash, check_password_hash  # Password hashing

app = Flask(__name__)  # Create Flask app
auth = HTTPBasicAuth()  # Initialize auth

# --- User storage: simple in-memory dictionary with hashed passwords ---
users = {
    "admin": generate_password_hash("admin123"),
}

# --- Verify password for HTTP Basic Auth ---
@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# --- Protect all modifying routes ---
def auth_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = auth.current_user()
        if not user:
            return auth.login_required(lambda: jsonify({'message': 'Authentication required'}))()
        return f(*args, **kwargs)
    return decorated

ROOT_DIR = os.path.abspath('files')  # Root folder for file storage
os.makedirs(ROOT_DIR, exist_ok=True)    # Ensure it exists

def safe_path(req_path):
    """Prevent directory traversal attacks."""
    full = os.path.abspath(os.path.join(ROOT_DIR, req_path))
    if not full.startswith(ROOT_DIR):
        raise Exception('Illegal path')
    return full

# Main page template: black background, red & gold accents, Bootstrap 5
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cloud Drive Manager</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    /* Page background and text colors */
    body {
      background-color: #000;        /* pure black */
      color: #e63946;                /* bright red */
      font-size: 1.2rem;
      user-select: none;
    }
    /* Gold highlight for buttons and hover */
    .text-gold { color: #ffc300 !important; }
    .btn-gold {
      background: none;
      border: 1px solid #ffc300;
      color: #ffc300;
      font-weight: 600;
    }
    .btn-gold:hover {
      background: #ffc300;
      color: #000;
    }
    /* File/folder list */
    ul { list-style: none; padding-left: 1.2rem; }
    li { padding: 0.2rem 0; cursor: pointer; }
    li.folder::before { content: "📁 "; }
    li.file::before   { content: "📄 "; }
    li.up { font-weight: bold; }
    li:hover { color: #ffc300; } /* gold on hover */
    /* Context menu */
    #contextMenu {
      position: absolute; background: #111;
      border: 1px solid #444; display: none;
      z-index: 1000; min-width: 140px;
    }
    #contextMenu div {
      padding: 0.5rem 1rem; cursor: pointer;
      color: #e63946;
    }
    #contextMenu div:hover { background: #222; color: #ffc300; }
    /* Drag-and-drop area */
    #dropZone {
      border: 2px dashed #444; padding: 1rem;
      text-align: center; margin-bottom: 1rem;
    }
    #dropZone.dragover { border-color: #ffc300; color: #ffc300; }
    /* Path display */
    #pathDisplay {
      font-weight: bold; margin-bottom: 1rem;
    }
  </style>
</head>
<body>
<div class="container-fluid py-3">
  <h1 class="text-gold mb-4">📂 Cloud Drive Manager</h1>

  <!-- Toolbar -->
  <div class="mb-3">
    <button class="btn btn-gold me-2" onclick="showRootMenu()">Menu</button>
    <input type="file" id="uploadInput" multiple style="display:none">
  </div>

  <!-- Current path -->
  <div id="pathDisplay"></div>

  <!-- Drag & Drop upload zone -->
  <div id="dropZone" class="mb-3 text-gold">
    Drag & Drop Files Here to Upload
  </div>

  <!-- File/Folder list -->
  <div id="fileList"></div>

  <!-- Custom context menu -->
  <div id="contextMenu"></div>
</div>

<!-- Edit modal for text files -->
<div class="modal fade" id="editModal" tabindex="-1">
  <div class="modal-dialog modal-lg">
    <div class="modal-content bg-dark text-gold">
      <div class="modal-header">
        <h5 class="modal-title">Edit: <span id="editFileName"></span></h5>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body p-0">
        <textarea id="editContent" class="form-control bg-black text-gold" rows="20"
                  style="font-family:monospace"></textarea>
      </div>
      <div class="modal-footer">
        <button id="saveBtn" class="btn btn-gold">Save</button>
        <button class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>

<!-- Bootstrap JS and custom script -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
/* Global variables */
let currentPath = '';
let ctxName = '', ctxType = '';
const ctxMenu = document.getElementById('contextMenu');
const uploadInput = document.getElementById('uploadInput');
const dropZone = document.getElementById('dropZone');
const editModal = new bootstrap.Modal(document.getElementById('editModal'));

/* List files via AJAX */
function listFiles(path='') {
  fetch('/list?path='+encodeURIComponent(path))
    .then(res => res.json())
    .then(data => {
      currentPath = path;
      document.getElementById('pathDisplay').textContent = 'Path: /' + path;
      let html = '<ul>';
      if (path) {
        const up = path.split('/').slice(0, -1).join('/');
        html += `<li class="up" onclick="listFiles('${up}')">⬆ Up</li>`;
      }
      data.forEach(item => {
        html += `<li class="${item.type}" 
                     oncontextmenu="showMenu(event,'${item.name}','${item.type}')" 
                     ondblclick="openItem('${item.name}','${item.type}')">
                   ${item.name}
                 </li>`;
      });
      html += '</ul>';
      document.getElementById('fileList').innerHTML = html;
    });
}

/* Open folder or download file */
function openItem(name, type) {
  const p = currentPath ? currentPath + '/' + name : name;
  if (type === 'folder') {
    listFiles(p);
  } else {
    window.open('/download?path=' + encodeURIComponent(p), '_blank');
  }
}

/* Show root menu */
function showRootMenu() {
  ctxName = ctxType = '';
  showCustomMenu([
    { text: 'New Folder', action: mkdir },
    { text: 'Upload File', action: () => uploadInput.click() }
  ], window.innerWidth/2, window.innerHeight/2);
}

/* Show context menu on right click */
function showMenu(e, name, type) {
  e.preventDefault();
  ctxName = name; ctxType = type;
  let items = [];
  if (type === 'folder') {
    items = [
      { text: 'Open', action: () => openItem(name, type) },
      { text: 'Rename', action: renameItem },
      { text: 'Delete', action: deleteItem },
      { text: 'Upload', action: () => uploadInput.click() },
      { text: 'New Folder', action: mkdir }
    ];
  } else {
    const ext = name.split('.').pop().toLowerCase();
    if (ext === 'txt') {
      items.push({ text: 'Edit', action: openEditor });
    } else if (['mp4','webm','ogg','jpg','png','gif'].includes(ext)) {
      items.push({ text: 'View', action: viewMedia });
    } else {
      items.push({ text: 'Download', action: () => openItem(name, type) });
    }
    items.push({ text: 'Rename', action: renameItem });
    items.push({ text: 'Delete', action: deleteItem });
  }
  showCustomMenu(items, e.pageX, e.pageY);
}

/* Render custom context menu */
function showCustomMenu(items, x, y) {
  ctxMenu.innerHTML = '';
  items.forEach(it => {
    const div = document.createElement('div');
    div.textContent = it.text;
    div.onclick = () => { it.action(); closeMenu(); };
    ctxMenu.appendChild(div);
  });
  ctxMenu.style.left = x + 'px';
  ctxMenu.style.top = y + 'px';
  ctxMenu.style.display = 'block';
  /* adjust if overflow */
  const rect = ctxMenu.getBoundingClientRect();
  if (rect.right > window.innerWidth) ctxMenu.style.left = (window.innerWidth - rect.width) + 'px';
  if (rect.bottom > window.innerHeight) ctxMenu.style.top = (window.innerHeight - rect.height) + 'px';
}
function closeMenu() { ctxMenu.style.display = 'none'; }
window.onclick = closeMenu; window.oncontextmenu = closeMenu;

/* Create folder AJAX */
function mkdir() {
  const name = prompt('Folder name');
  if (!name) return;
  fetch('/mkdir', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ path: currentPath, folder: name })
  }).then(r=>r.json()).then(d => {
    alert(d.message);
    listFiles(currentPath);
  });
}

/* Delete AJAX */
function deleteItem() {
  if (!ctxName || !confirm('Delete "'+ctxName+'"?')) return;
  const p = currentPath ? currentPath + '/' + ctxName : ctxName;
  fetch('/delete', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ path: p })
  }).then(r=>r.json()).then(d => {
    alert(d.message);
    listFiles(currentPath);
  });
}

/* Rename AJAX */
function renameItem() {
  if (!ctxName) return;
  const nn = prompt('New name', ctxName);
  if (!nn) return;
  const p = currentPath ? currentPath + '/' + ctxName : ctxName;
  fetch('/rename', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ path: p, new_name: nn })
  }).then(r=>r.json()).then(d => {
    alert(d.message);
    listFiles(currentPath);
  });
}

/* File upload AJAX */
uploadInput.onchange = () => {
  const files = uploadInput.files;
  if (!files.length) return;
  const fd = new FormData();
  for (let f of files) fd.append('files', f);
  fd.append('path', currentPath);
  fetch('/upload', { method:'POST', body: fd })
    .then(r=>r.json()).then(d => {
      alert(d.message);
      listFiles(currentPath);
      uploadInput.value = '';
    });
};

/* Drag & drop upload */
dropZone.ondragover = e => { e.preventDefault(); dropZone.classList.add('dragover'); };
dropZone.ondragleave = e => { e.preventDefault(); dropZone.classList.remove('dragover'); };
dropZone.ondrop = e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const fd = new FormData();
  for (let f of e.dataTransfer.files) fd.append('files', f);
  fd.append('path', currentPath);
  fetch('/upload', { method:'POST', body: fd })
    .then(r=>r.json()).then(d => {
      alert(d.message);
      listFiles(currentPath);
    });
};

/* Open text editor modal */
function openEditor() {
  const p = currentPath ? currentPath + '/' + ctxName : ctxName;
  document.getElementById('editFileName').textContent = ctxName;
  fetch('/edit?path='+encodeURIComponent(p))
    .then(r=>r.json()).then(d => {
      if (!d.ok) { alert(d.msg); return; }
      document.getElementById('editContent').value = d.content;
      editModal.show();
    });
}

/* Save text file AJAX */
document.getElementById('saveBtn').onclick = () => {
  const p = currentPath ? currentPath + '/' + ctxName : ctxName;
  const content = document.getElementById('editContent').value;
  fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ path: p, content: content })
  }).then(r=>r.json()).then(d => {
    alert(d.ok ? 'Saved' : d.msg);
    if (d.ok) editModal.hide();
  });
};

/* View media files */
function viewMedia() {
  const p = currentPath ? currentPath + '/' + ctxName : ctxName;
  window.open('/view?path='+encodeURIComponent(p), '_blank');
}

/* Initial load */
listFiles();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/list')
def list_files():
    req = request.args.get('path','').strip('/')
    try:
        d = safe_path(req)
        if not os.path.exists(d):
            return jsonify([])
        items = []
        for name in sorted(os.listdir(d), key=lambda x: (not os.path.isdir(os.path.join(d,x)), x.lower())):
            if name.startswith('.'): continue
            tp = 'folder' if os.path.isdir(os.path.join(d,name)) else 'file'
            items.append({'name': name, 'type': tp})
        return jsonify(items)
    except:
        return jsonify([])

@app.route('/download')
def download_file():
    req = request.args.get('path','').strip('/')
    try:
        full = safe_path(req)
        if not os.path.isfile(full):
            return "Not Found", 404
        directory, filename = os.path.dirname(full), os.path.basename(full)
        return send_from_directory(directory, filename, as_attachment=True)
    except:
        return "Forbidden", 403

@app.route('/delete', methods=['POST'])
@auth.login_required  # Require valid login for deletion
def delete_file_or_dir():
    data = request.get_json()
    req = data.get('path','').strip('/')
    if not req:
        return jsonify({'message':'Path empty'})
    try:
        full = safe_path(req)
        if not os.path.exists(full):
            return jsonify({'message':'Not exists'})
        if os.path.isdir(full):
            os.rmdir(full)
        else:
            os.remove(full)
        return jsonify({'message':'Deleted'})
    except Exception as e:
        return jsonify({'message': str(e)})

@app.route('/rename', methods=['POST'])
@auth.login_required  # Require login for renaming
def rename():
    data = request.get_json()
    req = data.get('path','').strip('/')
    new_name = data.get('new_name','').strip()
    if not new_name or '/' in new_name or '\\' in new_name:
        return jsonify({'message':'Illegal name'})
    try:
        old = safe_path(req)
        if not os.path.exists(old):
            return jsonify({'message':'Source not found'})
        new_full = os.path.join(os.path.dirname(old), new_name)
        if os.path.exists(new_full):
            return jsonify({'message':'Target exists'})
        os.rename(old, new_full)
        return jsonify({'message':'Renamed'})
    except Exception as e:
        return jsonify({'message': str(e)})

@app.route('/mkdir', methods=['POST'])
@auth.login_required  # Require login for mkdir
def mkdir():
    data = request.get_json()
    req = data.get('path','').strip('/')
    folder = data.get('folder','').strip()
    if not folder or '/' in folder or '\\' in folder:
        return jsonify({'message':'Illegal folder name'})
    try:
        parent = safe_path(req)
        newd = os.path.join(parent, folder)
        if os.path.exists(newd):
            return jsonify({'message':'Exists'})
        os.makedirs(newd)
        return jsonify({'message':'Created'})
    except Exception as e:
        return jsonify({'message': str(e)})

@app.route('/upload', methods=['POST'])
@auth.login_required  # Require login for upload
def upload():
    files = request.files.getlist('files')
    req = request.form.get('path','').strip('/')
    try:
        d = safe_path(req)
        if not os.path.exists(d):
            os.makedirs(d)
        count = 0
        for f in files:
            fn = os.path.basename(f.filename)
            if not fn or fn.startswith('.'): continue
            f.save(os.path.join(d, fn))
            count += 1
        return jsonify({'message': f'Uploaded {count} files'})
    except Exception as e:
        return jsonify({'message': str(e)})

@app.route('/edit')
@auth.login_required  # Require login to read text file for editing
def edit_file():
    req = request.args.get('path','').strip('/')
    try:
        full = safe_path(req)
        if not os.path.isfile(full) or not full.lower().endswith('.txt'):
            return jsonify({'ok': False, 'msg':'Not a text file'}), 400
        with open(full, encoding='utf-8') as f:
            content = f.read()
        return jsonify({'ok': True, 'content': content})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500

@app.route('/save', methods=['POST'])
@auth.login_required  # Require login to save edited text file
def save_file():
    data = request.get_json()
    req = data.get('path','').strip('/')
    content = data.get('content','')
    try:
        full = safe_path(req)
        if not os.path.isfile(full) or not full.lower().endswith('.txt'):
            return jsonify({'ok': False, 'msg':'Not a text file'}), 400
        with open(full, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500

VIEW_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><title>Viewer</title>
  <style>body{margin:0;background:#000;text-align:center;} img,video{max-width:100%;max-height:100vh;} </style>
</head>
<body>
  {% if is_video %}
    <video controls autoplay src="{{ url_for('download_file') }}?path={{ path|urlencode }}"></video>
  {% else %}
    <img src="{{ url_for('download_file') }}?path={{ path|urlencode }}">
  {% endif %}
</body>
</html>
"""

@app.route('/view')
def view_file():
    req = request.args.get('path','').strip('/')
    try:
        full = safe_path(req)
        if not os.path.isfile(full):
            abort(404)
        ext = os.path.splitext(full)[1].lower()
        is_video = ext in ('.mp4','.webm','.ogg')
        is_img   = ext in ('.jpg','.jpeg','.png','.gif','.bmp','.svg')
        if not (is_video or is_img):
            abort(400)
        return render_template_string(VIEW_HTML, is_video=is_video, path=req)
    except:
        abort(403)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
