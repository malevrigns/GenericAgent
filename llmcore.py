import os, json, re, time, requests, sys, threading, urllib3, base64, importlib, uuid, pathlib, copy
from datetime import datetime
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_RESP_CACHE_KEY = str(uuid.uuid4()); _RESP_CODEX_KEY = str(uuid.uuid4())
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path: sys.path.append(_ROOT)

def _load_mykeys():
    global _mykey_path
    try:
        sys.modules.pop('mykey', None)
        import mykey; _mykey_path = mykey.__file__
        return {k: v for k, v in vars(mykey).items() if not k.startswith('_')}
    except ImportError as e:
        if getattr(e, 'name', None) != 'mykey':
            raise Exception(f'[ERROR] mykey.py found but failed to import: {e}') from e
    except SyntaxError as e:
        raise Exception(f'[ERROR] mykey.py has syntax error: {e}') from e
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mykey.json')
    if not os.path.exists(p): raise Exception('[ERROR] mykey.py not found in sys.path and mykey.json not found. Run "python configure_mykey.py" or copy mykey_template.py to mykey.py and fill in your keys.')
    with open(_mykey_path := p, encoding='utf-8') as f: mk = json.load(f)
    if isinstance(mk, dict) and 'remote_url' in mk: return requests.get(mk['remote_url'], timeout=10).json()
    return mk

_mykey_lock = threading.Lock()
_mykey_path = _mykey_mtime = None
def reload_mykeys():
    global _mykey_mtime
    try:
        mt = os.stat(_mykey_path).st_mtime_ns if _mykey_path else -1
        if mt == _mykey_mtime: return globals().get('mykeys', {}), False
        with _mykey_lock: mk = _load_mykeys()
        _mykey_mtime = os.stat(_mykey_path).st_mtime_ns
        print(f'[Info] Load mykeys from {_mykey_path}')
        globals().update(mykeys=mk)
        return mk, True
    except: return globals().get('mykeys', {}), False

def __getattr__(name):  # once guard in PEP 562
    if name == 'mykeys': return reload_mykeys()[0]
    raise AttributeError(f"module 'llmcore' has no attribute {name}")

def compress_history_tags(messages, keep_recent=10, max_len=800, force=False, interval=5):
    """Compress <thinking>/<tool_use>/<tool_result> tags in older messages to save tokens."""
    compress_history_tags._cd = getattr(compress_history_tags, '_cd', 0) + 1
    if force: compress_history_tags._cd = 0
    if compress_history_tags._cd % interval != 0: return messages
    _before = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    _pats = {tag: re.compile(rf'(<{tag}>)([\s\S]*?)(</{tag}>)') for tag in ('thinking', 'think', 'tool_use', 'tool_result')}
    _hist_pat = re.compile(r'<(history|key_info|earlier_context)>[\s\S]*?</\1>')
    def _trunc_str(s): return s[:max_len//2] + '\n...[Truncated]...\n' + s[-max_len//2:] if isinstance(s, str) and len(s) > max_len else s
    def _trunc(text):
        text = _hist_pat.sub(lambda m: f'<{m.group(1)}>[...]</{m.group(1)}>', text)
        for pat in _pats.values(): text = pat.sub(lambda m: m.group(1) + _trunc_str(m.group(2)) + m.group(3), text)
        return text
    for i, msg in enumerate(messages):
        if i >= len(messages) - keep_recent: break
        c = msg['content']
        if isinstance(c, str): msg['content'] = _trunc(c)
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict): continue
                t = b.get('type')
                if t == 'text' and isinstance(b.get('text'), str): b['text'] = _trunc(b['text'])
                elif t == 'thinking' and isinstance(b.get('thinking'), str): b['thinking'] = _trunc_str(b['thinking'])
                elif t == 'tool_result':
                    tc = b.get('content')
                    if isinstance(tc, str): b['content'] = _trunc_str(tc)
                    elif isinstance(tc, list):
                        for sub in tc:
                            if isinstance(sub, dict) and sub.get('type') == 'text': sub['text'] = _trunc_str(sub.get('text'))
                elif t == 'tool_use' and isinstance(b.get('input'), dict):
                    for k, v in b['input'].items(): b['input'][k] = _trunc_str(v)
    print(f"[Cut] {_before} -> {sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)}")
    return messages

def _sanitize_leading_user_msg(msg):
    """把 user 消息里的 tool_result 块改写成纯文本，避免孤立引用。
    history 统一使用 Claude content-block 格式：content 是 list of blocks。"""
    msg = dict(msg)  # 浅拷贝外层 dict
    content = msg.get('content')
    if not isinstance(content, list): return msg
    texts = []
    for block in content:
        if not isinstance(block, dict): continue
        if block.get('type') == 'tool_result':
            c = block.get('content', '')
            if isinstance(c, list): texts.extend(b.get('text', '') for b in c if isinstance(b, dict))
            else: texts.append(str(c))
        elif block.get('type') == 'text': texts.append(block.get('text', ''))
    msg['content'] = [{"type": "text", "text": '\n'.join(t for t in texts if t)}]
    return msg

_oldprint = print
def safeprint(*argv):
    try: _oldprint(*argv)
    except OSError: pass
print = safeprint

STATS = {}

_CJK_RE = re.compile(r'[぀-ヿ⺀-鿿가-힯豈-﫿︰-﹏＀-￯]')
# 长段无空白 ASCII（base64/hex/长密钥等）：BPE 几乎合并不了，分词效率远低于普通英文
_DENSE_ASCII_RE = re.compile(r'[A-Za-z0-9+/=_\-]{64,}')

def estimate_tokens(text):
    """保守估算 token 数（无外部依赖）。仅作分词器缺失时的兜底：
    CJK/全角 ~1.2 字符/token（Qwen 中文实际 ~1.5，留安全边际），
    普通 ASCII ~3.5 字符/token，致密 ASCII 串(base64等) ~1.8 字符/token，
    emoji 等增补平面字符按 3 token/个。"""
    if not text: return 0
    cjk = len(_CJK_RE.findall(text))
    astral = sum(1 for ch in text if ord(ch) > 0xFFFF)
    dense = sum(len(m.group(0)) for m in _DENSE_ASCII_RE.finditer(text))
    normal = len(text) - cjk - astral - dense
    return int(cjk / 1.2 + normal / 3.5 + dense / 1.8 + astral * 3)

# ── 精确分词（彻底方案）：从模型服务器同步的同款 tokenizer.json，裁剪预算用真实 token 数 ──
# 字符估算在真实混合语料上实测偏低 12%（JSON转义stdout类内容偏低 20%），这正是
# 预防性裁剪"看起来没超窗、实际已爆"的根源。有精确分词器后预算零漂移。
_TOKENIZER = None
_TOKENIZER_TRIED = False
def _load_tokenizer():
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED: return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        d = os.environ.get('GA_TOKENIZER_DIR') or os.path.join(_ROOT, 'temp', 'qwen_tokenizer')
        p = os.path.join(d, 'tokenizer.json')
        if os.path.exists(p):
            from tokenizers import Tokenizer  # 可选依赖：pip install tokenizers；缺失则退回字符估算
            _TOKENIZER = Tokenizer.from_file(p)
            print(f'[Info] Exact tokenizer loaded: {p} (trim budgets now use real token counts)')
    except Exception as e:
        print(f'[WARN] Exact tokenizer unavailable, falling back to char estimate: {e}')
    return _TOKENIZER

def count_tokens(text):
    """有精确分词器则数真实 token，否则退回字符估算。"""
    if not text: return 0
    tok = _load_tokenizer()
    if tok is None: return estimate_tokens(text)
    return len(tok.encode(text, add_special_tokens=False).ids)

_MSG_TOK_CACHE = {}
def _msg_tokens(m):
    s = json.dumps(m, ensure_ascii=False)
    v = _MSG_TOK_CACHE.get(s)
    if v is None:
        v = count_tokens(s) + 10  # role/结构开销（chat template 每消息约 +4，10 留余量）
        if len(_MSG_TOK_CACHE) > 5000: _MSG_TOK_CACHE.clear()
        _MSG_TOK_CACHE[s] = v
    return v

def _clamp_oversized_blocks(history, token_budget, keep_from=0):
    """最后防线：逐条截断超大文本块；仍超预算则从最早的可删消息整体丢弃（至少留 2 条）。"""
    limit = max(256, token_budget // 6)  # 单块预算；按 1 token ≲ 1 字符保守截断
    def _cut(s):
        return s if len(s) <= limit else s[:limit//2] + '\n...[Truncated]...\n' + s[-limit//2:]
    for m in history:
        c = m.get('content')
        blocks = c if isinstance(c, list) else ([{'type': 'text', 'text': c}] if isinstance(c, str) else [])
        for b in blocks:
            if not isinstance(b, dict): continue
            for field in ('text', 'thinking'):
                if isinstance(b.get(field), str) and len(b[field]) > limit: b[field] = _cut(b[field])
            if b.get('type') == 'tool_result' and isinstance(b.get('content'), str) and len(b['content']) > limit:
                b['content'] = _cut(b['content'])
    while len(history) > 2 and sum(_msg_tokens(m) for m in history) > token_budget:
        del history[keep_from if keep_from < len(history) - 2 else 0]

_CTX_OVERFLOW_RE = re.compile(r'context_length_exceeded|maximum context length|max_model_len|exceeds?\s+the\s+maximum|maximum sequence length|longer than the specified maximum|context length|reduce the length', re.I)

def is_ctx_overflow_error(text):
    """判定服务端上下文超限错误（HTTP 400 等），用于触发强制压缩后重试。"""
    return isinstance(text, str) and text.lstrip().startswith(('!!!Error:', '[Error:')) and bool(_CTX_OVERFLOW_RE.search(text))

def trim_messages_history(history, sess, force=False):
    # 预算按真实 token 计（有 tokenizer 时精确计数；旧版按字符×3 曾把 262K 窗口撑爆，
    # 纯字符估算在混合语料上实测偏低 12%，依然会在 229K~239K 死亡区间漏判）
    # 服务端按 prompt+max_tokens 合计校验窗口（SGLang/vLLM 均如此），预算必须先扣回复额度；
    # 之前没扣，导致 context_win=240000+max_tokens=32768 超过 262144，落入 229K~239K 的请求必被 400
    reserve = min(getattr(sess, 'max_tokens', 0) or 0, sess.context_win // 4)
    cap = max(1024, sess.context_win - count_tokens(sess.system or '') - reserve - 1024)
    target = int(cap * (0.45 if force else getattr(sess, 'trim_keep_rate', 0.6)))
    kp = sess.trim_keep_prefix
    def cost(ms): return sum(_msg_tokens(m) for m in ms)
    compress_history_tags(history, interval=getattr(sess, 'cut_msg_interval', 7))
    STATS.update(ctx=(c := cost(history)), msgs=len(history)); print(f'[Debug] Current context: ~{c} tokens, {len(history)} messages.')
    if c <= cap and not force: return
    compress_history_tags(history, keep_recent=4, force=True)
    if cost(history) <= target and not force: return
    pre, post = history[:kp], history[kp:]; costs = [_msg_tokens(m) for m in post]; c = cost(pre) + sum(costs); i = 0
    # force 模式（服务端已判定超限、估算不再可信）：无视 target 直接砍到最少保留 9 条
    while len(post) - i > 9 and (c > target or force):
        c -= costs[i]; i += 1
        while i < len(post) and post[i].get('role') != 'user': c -= costs[i]; i += 1
        if i < len(post): old = costs[i]; post[i] = _sanitize_leading_user_msg(post[i]); costs[i] = _msg_tokens(post[i]); c += costs[i] - old
    post = post[i:]
    if kp and pre:
        m = pre[-1]
        if m.get('role') == 'assistant' and isinstance(m.get('content'), list):
            m['content'] = [b for b in m['content'] if not (isinstance(b, dict) and b.get('type') == 'tool_use')] or [{"type": "text", "text": "..."}]
        _d = lambda: [{"type": "text", "text": "..."}]
        gap = [{"role": "assistant", "content": _d()}] if m.get('role') == 'user' else [{"role": "user", "content": _d()}, {"role": "assistant", "content": _d()}]
        history[:] = pre + gap + post
    else: history[:] = pre + post
    if cost(history) > cap: _clamp_oversized_blocks(history, cap, keep_from=kp)
    STATS.update(ctx=(c := cost(history)), msgs=len(history)); print(f'[Debug] Trimmed context, current: ~{c} tokens, {len(history)} messages.')

def auto_make_url(base, path):
    b, p = base.rstrip('/'), path.strip('/')
    if b.endswith('$'): return b[:-1].rstrip('/')
    if b.endswith(p): return b
    return f"{b}/{p}" if re.search(r'/v\d+(/|$)', b) else f"{b}/v1/{p}"

def _parse_claude_json(data):
    if data.get("stop_reason") == "refusal":
        err = "[Error: Claude refusal]"
        yield err
        return [{"type": "text", "text": err}]
    content_blocks = data.get("content", [])
    _record_usage(data.get("usage", {}), "messages")
    for b in content_blocks:
        if b.get("type") == "text": yield b.get("text", "")
        elif b.get("type") == "thinking": yield ""
    return content_blocks

def _raise_if_retryable_overload(emsg):
    """HTTP 200 SSE/body overload → ConnectionError so _stream_with_retry can backoff."""
    if emsg and re.search(r'concurrency|retry later|overloaded|rate.?limit', emsg, re.I):
        raise requests.ConnectionError(emsg)

def _parse_claude_sse(resp_lines):
    """Parse Anthropic SSE stream. Yields text chunks, returns list[content_block]."""
    content_blocks = []; current_block = None; tool_json_buf = ""
    stop_reason = None; got_message_stop = False; warn = None
    for line in resp_lines:
        if not line: continue
        line = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line.startswith("data:"): continue
        data_str = line[5:].lstrip()
        if data_str == "[DONE]": break
        try: evt = json.loads(data_str)
        except Exception as e:
            print(f"[SSE] JSON parse error: {e}, line: {data_str[:200]}")
            continue
        evt_type = evt.get("type", "")
        if evt_type == "message_start":
            usage = evt.get("message", {}).get("usage", {})
            _record_usage(usage, "messages")
        elif evt_type == "content_block_start":
            block = evt.get("content_block", {})
            if block.get("type") == "text": current_block = {"type": "text", "text": ""}
            elif block.get("type") == "thinking": current_block = {"type": "thinking", "thinking": "", "signature": ""}
            elif block.get("type") == "tool_use":
                current_block = {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""), "input": {}}
                tool_json_buf = ""
        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if current_block and current_block.get("type") == "text": current_block["text"] += text
                if text: yield text
            elif delta.get("type") == "thinking_delta":
                thinking = delta.get("thinking", "")
                if current_block and current_block.get("type") == "thinking": current_block["thinking"] += thinking
                if thinking: yield thinking
            elif delta.get("type") == "signature_delta":
                if current_block and current_block.get("type") == "thinking":
                    current_block["signature"] = current_block.get("signature", "") + delta.get("signature", "")
            elif delta.get("type") == "input_json_delta": tool_json_buf += delta.get("partial_json", "")
        elif evt_type == "content_block_stop":
            if current_block:
                if current_block["type"] == "tool_use":
                    try: current_block["input"] = json.loads(tool_json_buf) if tool_json_buf else {}
                    except: current_block["input"] = {"_raw": tool_json_buf}
                content_blocks.append(current_block)
                current_block = None
        elif evt_type == "message_delta":
            delta = evt.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)
            out_usage = evt.get("usage", {})
            out_tokens = out_usage.get("output_tokens", 0)
            if out_tokens: STATS['out'] = out_tokens; print(f"[Output] tokens={out_tokens} stop_reason={stop_reason}")
        elif evt_type == "message_stop": got_message_stop = True
        elif evt_type == "error":
            err = evt.get("error", {})
            emsg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            _raise_if_retryable_overload(emsg)  # 走 _stream_with_retry，避免落到 ga 应用层
            warn = f"\n\n!!!Error: SSE {emsg}"; break
    if not warn:
        if not got_message_stop and not stop_reason: warn = "\n\n[!!! 流异常中断，未收到完整响应 !!!]"
        elif stop_reason == "max_tokens": warn = "\n\n[!!! Response truncated: max_tokens !!!]"
        elif stop_reason == "refusal": warn = "\n\n[Error: Claude refusal]"
    if current_block:
        if current_block["type"] == "tool_use":
            try: current_block["input"] = json.loads(tool_json_buf) if tool_json_buf else {}
            except: current_block["input"] = {"_raw": tool_json_buf}
        content_blocks.append(current_block); current_block = None
    if warn:
        print(f"[WARN] {warn.strip()}")
        insert_at = next((i for i,b in enumerate(content_blocks) if b.get("type") == "tool_use"), len(content_blocks))
        content_blocks.insert(insert_at, {"type": "text", "text": warn}); yield warn
    return content_blocks

def _try_parse_tool_args(raw):
    """Parse tool args string; split concatenated JSON objects like {..}{..} if needed.
    Returns list of parsed dicts."""
    if not raw: return [{}]
    try: return [json.loads(raw)]
    except: pass
    parts = re.split(r'(?<=\})(?=\{)', raw)
    if len(parts) > 1:
        parsed = []
        for p in parts:
            try: parsed.append(json.loads(p))
            except: return [{"_raw": raw}]
        return parsed
    return [{"_raw": raw}]

def _parse_openai_sse(resp_lines, api_mode="chat_completions"):
    """Parse OpenAI SSE stream (chat_completions or responses API).
    Yields text chunks, returns list[content_block].
    content_block: {type:'text', text:str} | {type:'tool_use', id:str, name:str, input:dict}
    """
    content_text = ""
    if api_mode == "responses":
        seen_delta = False; fc_buf = {}; current_fc_idx = None; reasoning_text = ""
        for line in resp_lines:
            if not line: continue
            line = line.decode('utf-8', errors='replace') if isinstance(line, bytes) else line
            if not line.startswith("data:"): continue
            data_str = line[5:].lstrip()
            if data_str == "[DONE]": break
            try: evt = json.loads(data_str)
            except: continue
            etype = evt.get("type", "")
            if etype == "response.output_text.delta":
                delta = evt.get("delta", "")
                if delta: seen_delta = True; content_text += delta; yield delta
            elif etype == "response.output_text.done" and not seen_delta:
                text = evt.get("text", "")
                if text: content_text += text; yield text
            elif etype == "response.reasoning_text.delta":
                delta = evt.get("delta", "")
                if delta: reasoning_text += delta
            elif etype == "response.reasoning_text.done":
                text = evt.get("text", "")
                if text: reasoning_text = text
            elif etype == "response.output_item.added":
                item = evt.get("item", {})
                if item.get("type") == "function_call":
                    idx = evt.get("output_index", 0)
                    fc_buf[idx] = {"id": item.get("call_id", item.get("id", "")), "name": item.get("name", ""), "args": ""}
                    current_fc_idx = idx
            elif etype == "response.function_call_arguments.delta":
                idx = evt.get("output_index", current_fc_idx or 0)
                if idx in fc_buf: fc_buf[idx]["args"] += evt.get("delta", "")
            elif etype == "response.function_call_arguments.done":
                idx = evt.get("output_index", current_fc_idx or 0)
                if idx in fc_buf: fc_buf[idx]["args"] = evt.get("arguments", fc_buf[idx]["args"])
            elif etype == "error":
                err = evt.get("error", {})
                emsg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                _raise_if_retryable_overload(emsg)
                if emsg: content_text += f"!!!Error: {emsg}"; yield f"!!!Error: {emsg}"
                break
            elif etype == "response.completed":
                usage = evt.get("response", {}).get("usage", {})
                _record_usage(usage, api_mode)
                break
            elif etype == "response.incomplete":
                # DeepSeek/OpenAI responses stream may end here (no [DONE]); treat as valid terminal,
                # record usage and surface a truncation marker instead of an empty-response retry storm.
                usage = (evt.get("response") or {}).get("usage", {})
                _record_usage(usage, api_mode)
                reason = ((evt.get("response") or {}).get("incomplete_details") or {}).get("reason", "") or "unknown"
                if not content_text and not fc_buf:
                    marker = f"[!!! output truncated: {reason}]"
                    content_text += marker; yield marker
                break
            elif etype == "response.failed":
                usage = (evt.get("response") or {}).get("usage", {})
                _record_usage(usage, api_mode)
                err = ((evt.get("response") or {}).get("error") or {})
                emsg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                _raise_if_retryable_overload(emsg)
                if emsg: content_text += f"!!!Error: {emsg}"; yield f"!!!Error: {emsg}"
                break
        blocks = []
        if reasoning_text: blocks.append({"type": "thinking", "thinking": reasoning_text})
        if content_text: blocks.append({"type": "text", "text": content_text})
        for idx in sorted(fc_buf):
            fc = fc_buf[idx]
            inps = _try_parse_tool_args(fc["args"])
            for i, inp in enumerate(inps):
                bid = fc["id"] or ''
                if len(inps) > 1: bid = f"{bid}_{i}" if bid else f"split_{i}"
                blocks.append({"type": "tool_use", "id": bid, "name": fc["name"], "input": inp})
        return blocks
    else:
        tc_buf = {}  # index -> {id, name, args}
        reasoning_text = ""
        for line in resp_lines:
            if not line: continue
            line = line.decode('utf-8', errors='replace') if isinstance(line, bytes) else line
            if not line.startswith("data:"): continue
            data_str = line[5:].lstrip()
            if data_str == "[DONE]": break
            try: evt = json.loads(data_str)
            except: continue
            ch = (evt.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if rc := delta.get("reasoning_content") or delta.get("reasoning", ""):
                reasoning_text += rc; yield rc
            if delta.get("content"):
                text = delta["content"]; content_text += text; yield text
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                has_name = bool(tc.get("function", {}).get("name"))
                if idx not in tc_buf:
                    if has_name or not tc_buf: tc_buf[idx] = {"id": tc.get("id") or '', "name": "", "args": ""}
                    else: idx = max(tc_buf)
                if has_name: tc_buf[idx]["name"] = tc["function"]["name"]
                if tc.get("function", {}).get("arguments"): tc_buf[idx]["args"] += tc["function"]["arguments"]
                if tc.get("id") and not tc_buf[idx]["id"]: tc_buf[idx]["id"] = tc["id"]
            usage = evt.get("usage")
            if usage: _record_usage(usage, api_mode)
            if ch.get("finish_reason") == "length":
                print("[WARN] Response truncated: max_tokens (思考+正文共享预算，被服务端截断)")
                content_text += "\n\n[!!! Response truncated: max_tokens !!!]"
        blocks = []
        if reasoning_text: blocks.append({"type": "thinking", "thinking": reasoning_text})
        if content_text: blocks.append({"type": "text", "text": content_text})
        for idx in sorted(tc_buf):
            tc = tc_buf[idx]
            inps = _try_parse_tool_args(tc["args"])
            for i, inp in enumerate(inps):
                bid = tc["id"] or ''
                if len(inps) > 1: bid = f"{bid}_{i}" if bid else f"split_{i}"
                blocks.append({"type": "tool_use", "id": bid, "name": tc["name"], "input": inp})
        return blocks

def _record_usage(usage, api_mode):
    if not usage: return
    # 上游常显式给 null；dict.get(k, 0) 遇 null 仍返回 None，后续加法会 TypeError
    def _i(v, default=0):
        try:
            return default if v is None else int(v)
        except (TypeError, ValueError):
            return default
    if api_mode == 'responses':
        cached = _i((usage.get("input_tokens_details") or {}).get("cached_tokens"))
        inp = _i(usage.get("input_tokens")); out = _i(usage.get("output_tokens"))
        print(f"[Cache] input={inp} cached={cached}")
        if out: print(f"[Output] tokens={out}")
    elif api_mode == 'chat_completions':
        cached = _i((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        inp = _i(usage.get("prompt_tokens")); out = _i(usage.get("completion_tokens"))
        print(f"[Cache] input={inp} cached={cached}")
        if out: print(f"[Output] tokens={out}")
    elif api_mode == 'messages':
        ci, cr, raw_inp = _i(usage.get("cache_creation_input_tokens")), _i(usage.get("cache_read_input_tokens")), _i(usage.get("input_tokens"))
        inp, cached, out = raw_inp + ci + cr, cr, 0
        print(f"[Cache] input={raw_inp} creation={ci} read={cr}")
    else: return
    STATS.update(inp=inp, cached=cached, out=out)
    
def _parse_openai_json(data, api_mode="chat_completions"):
    blocks = []
    if api_mode == "responses":
        _record_usage(data.get("usage") or {}, api_mode)
        for item in (data.get("output") or []):
            if item.get("type") == "message":
                for p in (item.get("content") or []):
                    if p.get("type") in ("output_text", "text") and p.get("text"):
                        blocks.append({"type": "text", "text": p["text"]}); yield p["text"]
            elif item.get("type") == "reasoning":
                for p in (item.get("content") or []):
                    if p.get("type") in ("reasoning_text", "summary_text") and p.get("text"):
                        blocks.append({"type": "thinking", "thinking": p["text"]})
            elif item.get("type") == "function_call":
                try: args = json.loads(item.get("arguments", "")) if item.get("arguments") else {}
                except: args = {"_raw": item.get("arguments", "")}
                blocks.append({"type": "tool_use", "id": item.get("call_id", item.get("id", "")),
                               "name": item.get("name", ""), "input": args})
        status = (data.get("status") or "").lower()
        if status == "failed":
            err = data.get("error") or {}
            emsg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            _raise_if_retryable_overload(emsg)
            if emsg: blocks.append({"type": "text", "text": f"!!!Error: {emsg}"}); yield f"!!!Error: {emsg}"
        elif status == "incomplete" and not any(b.get("type") == "text" for b in blocks):
            reason = ((data.get("incomplete_details") or {}).get("reason", "")) or "unknown"
            marker = f"[!!! output truncated: {reason}]"
            blocks.append({"type": "text", "text": marker}); yield marker
    else:
        _record_usage(data.get("usage") or {}, api_mode)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        reasoning = msg.get("reasoning_content") or msg.get("reasoning", "")
        if reasoning:
            blocks.append({"type": "thinking", "thinking": reasoning})
        content = msg.get("content", "")
        if content:
            blocks.append({"type": "text", "text": content}); yield content
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            try: args = json.loads(fn.get("arguments", "")) if fn.get("arguments") else {}
            except: args = {"_raw": fn.get("arguments", "")}
            blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": args})
    return blocks

def _stamp_oai_cache_markers(messages, model):
    """Add cache_control to last 2 user messages for Anthropic models via OAI-compatible relay."""
    ml = model.lower()
    if not any(k in ml for k in ('claude', 'anthropic')): return
    user_idxs = [i for i, m in enumerate(messages) if m.get('role') == 'user']
    for idx in user_idxs[-2:]:
        c = messages[idx].get('content')
        if isinstance(c, str):
            messages[idx] = {**messages[idx], 'content': [{'type': 'text', 'text': c, 'cache_control': {'type': 'ephemeral'}}]}
        elif isinstance(c, list) and c:
            c = list(c); c[-1] = dict(c[-1], cache_control={'type': 'ephemeral'})
            messages[idx] = {**messages[idx], 'content': c}

def _stream_with_retry(sess, url, headers, payload, parse_fn):
    STATS['session'] = sess.name
    _RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 529}
    cap = float(getattr(sess, 'max_retry_after', 60.0))
    def _delay(resp, attempt):
        try: ra = float((resp.headers or {}).get("retry-after"))
        except: ra = None
        return None if ra and ra > cap else max(0.5, ra or min(30.0, 3.0 * (2 ** attempt)))
    def _stopped(): return getattr(sess, 'should_stop', None) and sess.should_stop()
    def _sleep(d):  # interruptible sleep; True if aborted
        end = time.time() + d
        while time.time() < end:
            if _stopped(): return True
            time.sleep(0.2)
        return _stopped()
    for attempt in range(sess.max_retries + 1):
        if _stopped(): return []
        streamed = False
        STATS.update(t_start=time.time(), t_ttft=None)
        if not sess.stream: STATS['t_ttft'] = STATS['t_start']
        try:
            with requests.post(url, headers=headers, json=payload, stream=sess.stream, 
                               timeout=(sess.connect_timeout, sess.read_timeout), proxies=sess.proxies, verify=sess.verify) as r:
                sess.active_response = r
                if r.status_code >= 400:
                    #pathlib.Path(__file__).parent.joinpath('temp','bad_requests.json').write_text(json.dumps({"url":url,"headers":headers,"payload":payload,"t":time.time()},ensure_ascii=False),encoding='utf-8')
                    d = _delay(r, attempt) if r.status_code in _RETRYABLE and attempt < sess.max_retries else None
                    if d is not None:
                        print(f"[LLM Retry] HTTP {r.status_code}, retry in {d:.1f}s ({attempt+1}/{sess.max_retries+1})")
                        if _sleep(d): return []
                        continue
                    try: body = r.text.strip()[:500]
                    except: body = ""
                    err = f"!!!Error: HTTP {r.status_code}" + (f" (retry-after > {cap:.0f}s)" if d is None and r.status_code in _RETRYABLE and attempt < sess.max_retries else "") + (f": {body}" if body else "")
                    yield err; return [{"type": "text", "text": err}]
                gen = parse_fn(r)
                try:
                    while True:
                        if getattr(sess, 'should_stop', None) and sess.should_stop():
                            STATS['t_end'] = time.time(); return []
                        chunk = next(gen)
                        if chunk and STATS.get('t_ttft') is None: STATS['t_ttft'] = time.time()
                        streamed = True; yield chunk
                except StopIteration as e:
                    if not e.value and not streamed: raise requests.ConnectionError("empty response")
                    STATS['t_end'] = time.time()
                    STATS['tps'] = STATS.get('out', 0) / max(1e-9, STATS['t_end'] - max(STATS['t_ttft'] or 0, STATS['t_start']))
                    return e.value or []
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            err = f"!!!Error: {type(e).__name__}: {e}" if str(e) else f"!!!Error: {type(e).__name__}"
            if getattr(sess, 'should_stop', None) and sess.should_stop(): return []
            if attempt < sess.max_retries:
                d = _delay(None, attempt)
                print(f"[LLM Retry] {type(e).__name__}, retry in {d:.1f}s ({attempt+1}/{sess.max_retries+1})")
                if _sleep(d): return []
                continue
            yield err; return [{"type": "text", "text": err}]
        except Exception as e:
            err = f"\n\n[!!! 流异常中断 {type(e).__name__}: {e} !!!]" if streamed else f"!!!Error: {type(e).__name__}: {e}"
            yield err; return [{"type": "text", "text": err}]

def _openai_stream(sess, messages):
    model, api_mode = sess.model, sess.api_mode
    ml = model.lower()
    temperature = sess.temperature
    if 'kimi' in ml or 'moonshot' in ml: temperature = 1
    elif 'minimax' in ml: temperature = max(0.01, min(temperature, 1.0))  # MiniMax requires temp in (0, 1]
    headers = {"Authorization": f"Bearer {sess.api_key}", "Content-Type": "application/json", "Accept": "text/event-stream", 'originator': 'codex_exec'}
    headers["User-Agent"] = sess.user_agent
    if api_mode == "responses":
        url = auto_make_url(sess.api_base, "responses")
        payload = {"model": model, "input": _to_responses_input(messages), "stream": sess.stream, 
                   "prompt_cache_key": _RESP_CACHE_KEY, "instructions": sess.system or "You are an Omnipotent Executor.",
                   "client_metadata": {"x-codex-window-id": f"{_RESP_CACHE_KEY}:0","x-codex-installation-id": _RESP_CODEX_KEY},
                   'include': ['reasoning.encrypted_content']}
        if sess.reasoning_effort: payload["reasoning"] = {"effort": sess.reasoning_effort}
        if sess.max_tokens: payload["max_output_tokens"] = sess.max_tokens
    else:
        url = auto_make_url(sess.api_base, "chat/completions")
        if sess.system: messages = [{"role": "system", "content": sess.system}] + messages
        _stamp_oai_cache_markers(messages, model)
        payload = {"model": model, "messages": messages, "stream": sess.stream}
        if sess.stream: payload["stream_options"] = {"include_usage": True}
        if temperature != 1: payload["temperature"] = temperature
        if getattr(sess, 'top_p', None) is not None: payload["top_p"] = sess.top_p
        if getattr(sess, 'top_k', None) is not None: payload["top_k"] = sess.top_k
        if getattr(sess, 'repetition_penalty', None) is not None: payload["repetition_penalty"] = sess.repetition_penalty
        if sess.max_tokens: payload["max_completion_tokens" if ml.startswith(("gpt-5", "o1", "o2", "o3", "o4")) else "max_tokens"] = sess.max_tokens
        if sess.reasoning_effort: payload["reasoning_effort"] = sess.reasoning_effort
        if getattr(sess, 'chat_template_kwargs', None): payload["chat_template_kwargs"] = sess.chat_template_kwargs
    tools = getattr(sess, 'tools', None)
    if tools: payload["tools"] = _prepare_oai_tools(tools, api_mode)
    if sess.service_tier: payload["service_tier"] = sess.service_tier
    parse_fn = (lambda r: _parse_openai_sse(r.iter_lines(), api_mode)) if sess.stream else (lambda r: _parse_openai_json(r.json(), api_mode))
    return (yield from _stream_with_retry(sess, url, headers, payload, parse_fn))
        
def _prepare_oai_tools(tools, api_mode="chat_completions"):
    if api_mode == "responses":
        resp_tools = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                rt = {"type": "function"}; rt.update(t["function"])
                resp_tools.append(rt)
            else: resp_tools.append(t)
        return resp_tools
    return tools

def _to_responses_input(messages):
    result, pending = [], []
    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        if role == "tool":
            cid = msg.get("tool_call_id") or (pending.pop(0) if pending else f"call_{uuid.uuid4().hex[:8]}")
            result.append({"type": "function_call_output", "call_id": cid, "output": msg.get("content", "")})
            continue
        if role not in ["user", "assistant", "system", "developer"]: role = "user"
        if role == "system": role = "developer"  # Responses API uses 'developer' instead of 'system'
        content = msg.get("content", "")
        text_type = "output_text" if role == "assistant" else "input_text"
        parts = []
        if isinstance(content, str):
            if content: parts.append({"type": text_type, "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict): continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text", "")
                    if text: parts.append({"type": text_type, "text": text})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url and role != "assistant": parts.append({"type": "input_image", "image_url": url})
        if len(parts) == 0: parts = [{"type": text_type, "text": str(content) if not isinstance(content, list) else '[empty]'}]
        result.append({"role": role, "content": parts})
        pending = []
        for tc in (msg.get("tool_calls") or []):
            f = tc.get("function", {})
            cid = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            pending.append(cid)
            result.append({"type": "function_call", "call_id": cid, "name": f.get("name", ""), "arguments": f.get("arguments", "")})
    return result


def _msgs_claude2oai(messages):
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        if role == "assistant":
            text_parts, tool_calls, reasoning = [], [], ""
            for b in blocks:
                if not isinstance(b, dict): continue
                if b.get("type") == "thinking" and b.get("thinking"): reasoning = b["thinking"]
                elif b.get("type") == "text" and b.get("text"): text_parts.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b.get("id") or '', "type": "function",
                        "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}
                    })
            m = {"role": "assistant"}
            if reasoning: m["reasoning_content"] = reasoning
            if text_parts: m["content"] = text_parts
            elif not tool_calls: m["content"] = "."
            if tool_calls: m["tool_calls"] = tool_calls
            result.append(m)
        elif role == "user":
            text_parts = []
            for b in blocks:
                if not isinstance(b, dict): continue
                if b.get("type") == "tool_result":
                    if text_parts:
                        result.append({"role": "user", "content": text_parts})
                        text_parts = []
                    tr = b.get("content", "")
                    if isinstance(tr, list):
                        tr = "\n".join(x.get("text", "") for x in tr if isinstance(x, dict) and x.get("type") == "text")
                    result.append({"role": "tool", "tool_call_id": b.get("tool_use_id") or '', "content": tr if isinstance(tr, str) else str(tr)})
                elif b.get("type") == "image":
                    src = b.get("source") or {}
                    if src.get("type") == "base64" and src.get("data"):
                        text_parts.append({"type": "image_url", "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}})
                elif b.get("type") == "image_url": text_parts.append(b)
                elif b.get("type") == "text" and b.get("text"): text_parts.append({"type": "text", "text": b.get("text", "")})
            if text_parts: result.append({"role": "user", "content": text_parts})
        else: result.append(msg)
    return result


class BaseSession:
    def __init__(self, cfg):
        self.api_key = cfg['apikey']
        self.api_base = cfg['apibase'].rstrip('/')
        self.model = cfg.get('model', '')
        default_context_win = 35000; default_cut_msg_interval = 7
        if 'deepseek' in self.model.lower():
            default_context_win = 80000; default_cut_msg_interval = 25; self.trim_keep_rate = 0.3
        self.context_win = cfg.get('context_win', default_context_win)
        self.maxlen_multiplier = min(max(self.context_win / default_context_win * 0.75, 1.0), 3.0)
        self.cut_msg_interval = int(default_cut_msg_interval * self.maxlen_multiplier)
        self.trim_keep_prefix = max(0, int(cfg.get('trim_keep_prefix', 0) or 0))
        self.history = []; self.lock = threading.Lock(); self.system = ""
        self.name = cfg.get('name', self.model)
        self.extra_sys_prompt = cfg.get('extra_sys_prompt', '')
        if cfg.get('extra_sys_prompt_file'):
            self.extra_sys_prompt = (self.extra_sys_prompt or '') + open(cfg['extra_sys_prompt_file'] if os.path.isabs(cfg['extra_sys_prompt_file']) else os.path.join(_ROOT, cfg['extra_sys_prompt_file']), encoding='utf-8').read()
        proxy = cfg.get('proxy'); 
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        self.max_retries = max(0, int(cfg.get('max_retries', 4)))
        self.max_retry_after = float(cfg.get('max_retry_after', 60.0))
        self.verify = cfg.get('verify', True)
        self.stream = cfg.get('stream', True)
        default_ct, default_rt = (5, 40) if self.stream else (10, 240)
        self.connect_timeout = max(1, int(cfg.get('timeout', default_ct)))
        self.read_timeout = max(5, int(cfg.get('read_timeout', default_rt)))
        def _enum(key, valid):
            v = cfg.get(key); v = None if v is None else str(v).strip().lower()
            return v if not v or v in valid else print(f"[WARN] Invalid {key} {v!r}, ignored.")
        self.reasoning_effort = _enum('reasoning_effort', {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'})
        self.service_tier = _enum('service_tier', {'auto', 'default', 'priority', 'flex'})
        self.thinking_type = _enum('thinking_type', {'adaptive', 'enabled', 'disabled'})
        self.thinking_budget_tokens = cfg.get('thinking_budget_tokens')
        self.omit_thinking = cfg.get('omit_thinking', False)  # Exclude thinking from session history
        mode = str(cfg.get('api_mode', 'chat_completions')).strip().lower().replace('-', '_')
        self.api_mode = 'responses' if mode in ('responses', 'response') else 'chat_completions'
        self.temperature = cfg.get('temperature', 1)
        self.top_p = cfg.get('top_p')
        self.top_k = cfg.get('top_k')
        self.repetition_penalty = cfg.get('repetition_penalty')
        self.max_tokens = cfg.get('max_tokens')
        self.chat_template_kwargs = cfg.get('chat_template_kwargs')  # e.g. {'enable_thinking': True} for SGLang-hosted Qwen
        self.default_ua = "claude-cli/2.1.152 (external, cli)"
        self.user_agent = cfg.get("user_agent", self.default_ua)
    def _apply_claude_thinking(self, payload):
        if self.thinking_type:
            thinking = {"type": self.thinking_type}
            if self.thinking_type == 'enabled':
                if self.thinking_budget_tokens is None: print("[WARN] thinking_type='enabled' requires thinking_budget_tokens, ignored.")
                else:
                    thinking["budget_tokens"] = self.thinking_budget_tokens; payload["thinking"] = thinking
            else: payload["thinking"] = thinking
        if self.reasoning_effort:
            effort = {'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'max', 'max': 'max'}.get(self.reasoning_effort)
            if effort: payload["output_config"] = {"effort": effort}
            else: print(f"[WARN] reasoning_effort {self.reasoning_effort!r} is unsupported for Claude output_config.effort, ignored.")
    def ask(self, prompt):
        def _ask_gen():
            with self.lock:
                self.history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                trim_messages_history(self.history, self)
                messages = self.make_messages(self.history)
            content_blocks = None; content = ''
            for _attempt in (1, 2):
                gen = self.raw_ask(messages)
                try: first = next(gen)
                except StopIteration as e: first, content_blocks = None, (e.value or [])
                if _attempt == 1 and (is_ctx_overflow_error(first) or (content_blocks and is_ctx_overflow_error(next((b.get('text') for b in content_blocks if isinstance(b, dict) and b.get('type') == 'text'), '')))):
                    # 服务端判定上下文超限（估算偏低时预防性裁剪可能没触发）：强制深度压缩后透明重试一次
                    print("[Context Guard] 上下文超限被服务端拒绝，强制压缩历史后重试。")
                    with self.lock:
                        trim_messages_history(self.history, self, force=True)
                        messages = self.make_messages(self.history)
                    content_blocks = None; content = ''
                    continue
                if first: content += first; yield first
                try:
                    while True: chunk = next(gen); content += chunk; yield chunk
                except StopIteration as e: content_blocks = e.value or []
                break
            if len(content_blocks) > 1: print(f"[DEBUG BaseSession.ask] content_blocks: {content_blocks}")
            for block in (content_blocks or []):
                if block.get('type', '') == 'tool_use':
                    tu = {'name': block.get('name', ''), 'arguments': block.get('input', {})}
                    yield f'<tool_use>{json.dumps(tu, ensure_ascii=False)}</tool_use>'
            if content.strip() and not content.startswith("!!!Error:"): self.history.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        return _ask_gen()

def _keep_claude_block(b): return not isinstance(b, dict) or b.get("type") != "thinking" or b.get("signature")
def _drop_unsigned_thinking(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, list): m["content"] = [b for b in c if _keep_claude_block(b)]
    return messages

def _ensure_thinking_blocks(messages, model):
    """deepseek needs thinking in history!"""
    if 'deepseek' not in model.lower(): return messages
    for m in messages:
        if m.get("role") != "assistant": continue
        c = m.get("content")
        if not isinstance(c, list): continue
        has_thinking = any(isinstance(b, dict) and b.get("type") == "thinking" for b in c)
        if not has_thinking: m["content"] = [{"type": "thinking", "thinking": "...", "signature": "placeholder"}, *c]
    return messages

class ClaudeSession(BaseSession):
    def raw_ask(self, messages):
        messages = _fix_messages(messages)
        if self.max_tokens is None: self.max_tokens = 8192
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01", "anthropic-beta": "prompt-caching-2024-07-31"}
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "stream": self.stream}
        if self.temperature != 1: payload["temperature"] = self.temperature
        self._apply_claude_thinking(payload)
        if self.system: payload["system"] = [{"type": "text", "text": self.system, "cache_control": {"type": "persistent"}}]
        url = auto_make_url(self.api_base, "messages")
        parse_fn = (lambda r: _parse_claude_sse(r.iter_lines())) if self.stream else (lambda r: _parse_claude_json(r.json()))
        return (yield from _stream_with_retry(self, url, headers, payload, parse_fn))
    def make_messages(self, raw_list):
        msgs = _drop_unsigned_thinking([{"role": m['role'], "content": list(m['content'])} for m in raw_list])
        user_idxs = [i for i, m in enumerate(msgs) if m['role'] == 'user']
        for idx in user_idxs[-2:]:
            msgs[idx]["content"][-1] = dict(msgs[idx]["content"][-1], cache_control={"type": "ephemeral"})
        return msgs

class LLMSession(BaseSession):
    def raw_ask(self, messages): return (yield from _openai_stream(self, messages))
    def make_messages(self, raw_list): return _msgs_claude2oai(_fix_messages(raw_list))

def _fix_messages(messages):
    if not messages: return messages
    W = lambda c: c if isinstance(c, list) else [{"type": "text", "text": str(c)}]
    merged = []
    for m in messages:
        if m.get('role') not in ('user', 'assistant'): continue
        blocks = W(m.get('content', []))
        if merged and m['role'] == merged[-1]['role']:
            merged[-1]['content'] += list(blocks)
        else:
            merged.append({"role": m['role'], "content": list(blocks)})
    while merged and merged[0]['role'] != 'user': merged.pop(0)
    if not merged: return []
    prev_uses = []
    for m in merged:
        c = m['content']
        if m['role'] == 'assistant':
            seen, out = set(), []
            for b in c:
                uid = b.get('id') if isinstance(b, dict) and b.get('type') == 'tool_use' else None
                if uid and uid in seen: continue
                if uid: seen.add(uid)
                out.append(b)
            m['content'] = out
            prev_uses = [b.get('id') for b in out if isinstance(b, dict) and b.get('type') == 'tool_use']
        else:
            got, rest = {}, []
            for b in c:
                tid = b.get('tool_use_id') if isinstance(b, dict) and b.get('type') == 'tool_result' else None
                if tid and tid in prev_uses and tid not in got: got[tid] = b
                elif isinstance(b, dict) and b.get('type') == 'tool_result': rest.append({"type": "text", "text": str(b.get('content', ''))})
                else: rest.append(b)
            m['content'] = [got.get(u) or {"type": "tool_result", "tool_use_id": u, "content": "(error)"} for u in prev_uses] + rest
            prev_uses = []
    for m in merged: m['content'] = [b for b in m['content'] if not (isinstance(b, dict) and b.get('type') == 'text' and not (b.get('text') or '').strip())] or [{"type": "text", "text": "."}]
    return merged

class NativeClaudeSession(BaseSession):
    native_ua = "claude-cli/2.1.152 (native, cli)"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fake_cc_system_prompt = cfg.get("fake_cc_system_prompt", False)
        self._session_id = str(uuid.uuid4())
        self._account_uuid = str(uuid.uuid4())
        self._device_id = uuid.uuid4().hex + uuid.uuid4().hex[:32]
        self.tools = None
        if self.user_agent == self.default_ua: self.user_agent = self.native_ua
        self.api_key_header = str(cfg.get('api_key_header', 'auto')).strip().lower()
    def raw_ask(self, messages):
        if self.max_tokens is None: self.max_tokens = 8192
        model = self.model
        messages = _fix_messages(messages)
        if 'claude' in model.lower(): messages = _drop_unsigned_thinking(messages)
        messages = _ensure_thinking_blocks(messages, self.model)
        beta_parts = ["claude-code-20250219", "interleaved-thinking-2025-05-14", "redact-thinking-2026-02-12", "thinking-token-count-2026-05-13", "context-management-2025-06-27", "prompt-caching-scope-2026-01-05", "mid-conversation-system-2026-04-07", "effort-2025-11-24", "fallback-credit-2026-06-01"]
        if "[1m]" in model.lower():
            beta_parts.insert(1, "context-1m-2025-08-07"); model = model.replace("[1m]", "").replace("[1M]", "")
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
            "anthropic-beta": ",".join(beta_parts), "anthropic-dangerous-direct-browser-access": "true",
            "user-agent": self.user_agent, "x-app": "cli"}
        headers.update({"Accept": "application/json", "X-Claude-Code-Session-Id": self._session_id, "X-Stainless-Arch": "x64", "X-Stainless-Lang": "js", "X-Stainless-OS": "Windows", "X-Stainless-Package-Version": "0.94.0", "X-Stainless-Retry-Count": "0", "X-Stainless-Runtime": "node", "X-Stainless-Runtime-Version": "v24.3.0", "X-Stainless-Timeout": "600"})
        if self.api_key_header == 'x-api-key': headers["x-api-key"] = self.api_key
        elif self.api_key_header == 'bearer': headers["authorization"] = f"Bearer {self.api_key}"
        elif self.api_key.startswith("sk-ant-"): headers["x-api-key"] = self.api_key
        else: headers["authorization"] = f"Bearer {self.api_key}"
        payload = {"model": model, "messages": messages, "max_tokens": self.max_tokens, "stream": self.stream}
        #if self.fake_cc_system_prompt: payload["max_tokens"] = 64000
        if self.temperature != 1: payload["temperature"] = self.temperature
        self._apply_claude_thinking(payload)
        #payload["context_management"] = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}; 
        if self.fake_cc_system_prompt:
            if 'thinking' not in payload: payload["thinking"] = {"type": "adaptive"}
            if 'output_config' not in payload: payload["output_config"] = {"effort": "medium"}
        payload["metadata"] = {"user_id": json.dumps({"device_id": self._device_id, "account_uuid": "", "session_id": self._session_id}, separators=(',', ':'))}
        if self.tools:
            claude_tools = openai_tools_to_claude(self.tools)
            tools = [dict(t) for t in claude_tools]; tools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = tools
        else: print("[ERROR] No tools provided for this session.")
        payload['system'] = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude.", "cache_control": {"type": "ephemeral"}}]
        #payload['system'][0]['text'] += f"\nPlatform: {sys.platform}"
        if self.system:
            if self.fake_cc_system_prompt: payload["system"].append({"type": "text", "text": self.system})
            else: payload["system"] = [{"type": "text", "text": self.system}]
        user_idxs = [i for i, m in enumerate(messages) if m['role'] == 'user']
        for idx in user_idxs[-2:]:
            messages[idx] = {**messages[idx], "content": list(messages[idx]["content"])}
            messages[idx]["content"][-1] = dict(messages[idx]["content"][-1], cache_control={"type": "ephemeral"})
        url = auto_make_url(self.api_base, "messages") + '?beta=true'
        parse_fn = (lambda r: _parse_claude_sse(r.iter_lines())) if self.stream else (lambda r: _parse_claude_json(r.json()))
        return (yield from _stream_with_retry(self, url, headers, payload, parse_fn))

    def ask(self, msg):
        assert type(msg) is dict
        with self.lock:
            self.history.append(msg)
            trim_messages_history(self.history, self)
            messages = [{"role": m["role"], "content": list(m["content"])} for m in self.history]
        content_blocks = None
        for _attempt in (1, 2):
            gen = self.raw_ask(messages)
            try: first = next(gen)
            except StopIteration as e: first, content_blocks = None, (e.value or [])
            if _attempt == 1 and (is_ctx_overflow_error(first) or (content_blocks and is_ctx_overflow_error(next((b.get('text') for b in content_blocks if isinstance(b, dict) and b.get('type') == 'text'), '')))):
                # 服务端判定上下文超限：强制深度压缩后透明重试一次（错误块不流出、不入历史）
                print("[Context Guard] 上下文超限被服务端拒绝，强制压缩历史后重试。")
                with self.lock:
                    trim_messages_history(self.history, self, force=True)
                    messages = [{"role": m["role"], "content": list(m["content"])} for m in self.history]
                content_blocks = None
                continue
            if first: yield first
            try:
                while True: yield next(gen)
            except StopIteration as e: content_blocks = e.value or []
            break
        if content_blocks and (_injected := _ensure_text_block(content_blocks)): yield _injected
        if content_blocks and not (len(content_blocks) == 1 and content_blocks[0].get("text", "").startswith("!!!Error:")):
            history_blocks = content_blocks
            if self.omit_thinking: history_blocks = [b for b in content_blocks if b.get("type") != "thinking"]
            self.history.append({"role": "assistant", "content": history_blocks})
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        content = "\n".join(text_parts).strip()
        tool_calls = [MockToolCall(b["name"], b.get("input", {}), id=b.get("id", "")) for b in content_blocks if b.get("type") == "tool_use"]
        if not tool_calls: tool_calls, content = _parse_text_tool_calls(content)
        thinking_parts = [b["thinking"] for b in content_blocks if b.get("type") == "thinking"]
        thinking = "\n".join(thinking_parts).strip()
        if not thinking:
            think_pattern = r"<think(?:ing)?>(.*?)</think(?:ing)?>"
            think_match = re.search(think_pattern, content, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                content = re.sub(think_pattern, "", content, flags=re.DOTALL)
        raw = "[" + ",\n".join(repr(b) for b in content_blocks) + "]"
        return MockResponse(thinking, content, tool_calls, raw)

class NativeOAISession(NativeClaudeSession):
    native_ua = "codex_exec/0.139.0 (Windows 10.0.26200; x86_64) unknown (codex_exec; 0.139.0)"
    def raw_ask(self, messages):
        messages = _fix_messages(messages)
        messages = _ensure_thinking_blocks(messages, self.model)
        return (yield from _openai_stream(self, _msgs_claude2oai(messages)))

def openai_tools_to_claude(tools):
    """[{type:'function', function:{name,description,parameters}}] → [{name,description,input_schema}]."""
    result = []
    for t in tools:
        if 'input_schema' in t: result.append(t); continue  # 已是claude格式
        fn = t.get('function', t)
        result.append({'name': fn['name'], 'description': fn.get('description', ''),
            'input_schema': fn.get('parameters', {'type': 'object', 'properties': {}})})
    return result

class MockFunction:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments  
         
class MockToolCall:
    def __init__(self, name, args, id=''):
        arg_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else (args or '{}')
        self.function = MockFunction(name, arg_str); self.id = id

class MockResponse:
    def __init__(self, thinking, content, tool_calls, raw, stop_reason='end_turn'):
        self.thinking = thinking; self.content = content          
        self.tool_calls = tool_calls; self.raw = raw
        self.stop_reason = 'tool_use' if tool_calls else stop_reason
    def __repr__(self):    
        return f"<MockResponse thinking={bool(self.thinking)}, content='{self.content}', tools={bool(self.tool_calls)}>"

class ToolClient:
    def __init__(self, backend, auto_save_tokens=True):
        self.backend = backend
        self.auto_save_tokens = auto_save_tokens
        self.last_tools = ''
        self.name = self.backend.name
        self.total_cd_tokens = 0
        self.log_path = None

    def chat(self, messages, tools=None):
        tools = json.loads(json.dumps(tools, ensure_ascii=False)) if tools else tools
        for t in tools or []:
            f = t.get('function', {})
            if f.get('name') == 'file_write':
                props = f.get('parameters', {}).get('properties', {})
                props.pop('content', None)
                extra = '. Content must be placed in <file_content> tags in reply body, not in args'
                if extra not in f.get('description', ''): f['description'] = f.get('description', '') + extra
                break
        full_prompt = self._build_protocol_prompt(messages, tools)
        print("Full prompt length:", len(full_prompt), 'chars')
        gen = self.backend.ask(full_prompt)
        _write_llm_log('Prompt', full_prompt, self.log_path)
        raw_text = ''
        # 流式循环检测：服务器端 repetition_penalty 在部分构建里不生效（实测 vendor PPU build
        # 丢弃 rep/freq/presence penalty），思考型小模型会陷入复读直到 max_tokens。此处发现
        # 同一长片段在近期输出里反复出现即中止流，让上层按错误重试。
        # 阈值放宽：长任务（批量网页操作、代码生成）中合法输出也常重复相似结构，
        # 需要更大的窗口、更长的重复单元、更多重复次数和更高覆盖率才判定为失控复读。
        _LOOP_WIN, _LOOP_MIN, _LOOP_DUP, _LOOP_COVER = 600, 80, 5, 0.85
        _recent = ''; _scanned_at = 0; _abort = None
        for chunk in gen:
            raw_text += chunk
            _recent = (_recent + chunk)[-_LOOP_WIN * 2:]
            if len(_recent) > _LOOP_WIN and len(_recent) - _scanned_at >= 24:
                _scanned_at = len(_recent)
                probe = _recent[-_LOOP_WIN:]
                for L in range(_LOOP_MIN, min(len(_recent) // 3, 240) + 1):
                    unit = _recent[-L:]
                    cnt = _recent.count(unit)
                    if cnt >= _LOOP_DUP and cnt * L >= _LOOP_COVER * len(probe):
                        _abort = (unit[:80], L, cnt); break
                if _abort: break
            yield chunk
        if _abort:
            unit, L, cnt = _abort
            print(f"[WARN] Repetition loop detected (unit {L} chars x{cnt}): {unit!r} — aborting stream for retry")
            _write_llm_log('Response(loop-aborted)', raw_text, self.log_path, model=self.backend.model)
            # 合成一条带错误标记的回复：do_no_tool 的 '!!!Error:' 检测会触发强制重试而非终局
            return MockResponse('', f"[Repetition loop aborted: 同一片段重复{cnt}次，服务器惩罚参数失效被流式检测截断。 !!!Error: repetition loop]", [], raw_text)
        _write_llm_log('Response', raw_text, self.log_path, model=self.backend.model)
        return self._parse_mixed_response(raw_text)

    def _prepare_tool_instruction(self, tools):
        tool_instruction = ""
        if not tools: return tool_instruction
        tools_json = json.dumps(tools, ensure_ascii=False, separators=(',', ':'))
        _en = os.environ.get('GA_LANG') == 'en'
        if _en:
            tool_instruction = f"""
### Interaction Protocol (must follow strictly, always in effect)
Follow these steps to think and act:
1. **Think**: Analyze the current situation and strategy inside `<thinking>` tags.
2. **Summarize**: Output a minimal one-line (<30 words) physical snapshot in `<summary>`: new info from last tool result + current tool call intent. This goes into long-term working memory. Must contain real information, no filler.
3. **Act**: If you need to call tools, output one or more **<tool_use> blocks** after your reply, then stop.
"""
        else:
            tool_instruction = f"""
### 交互协议 (必须严格遵守，持续有效)
请按照以下步骤思考并行动：
1. **思考**: 在 `<thinking>` 标签中先进行思考，分析现状和策略。
2. **总结**: 在 `<summary>` 中输出*极为简短*的高度概括的单行（<30字）物理快照，包括上次工具调用结果产生的新信息+本次工具调用意图。此内容将进入长期工作记忆，记录关键信息，严禁输出无实际信息增量的描述。
3. **行动**: 如需调用工具，请在回复正文之后输出一个（或多个）**<tool_use>块**，然后结束。
"""
        if _en:
            tool_instruction += (
                '\nFormat: ```<tool_use>{"name": "tool_name", "arguments": {...}}</tool_use>```\n'
                'Hard output requirements (violations break the parser and abort the task):\n'
                '- Each <tool_use> block contains exactly ONE complete JSON object; before finishing, verify every { has a matching }, every [ a matching ], strings use double quotes, and literal newlines inside strings are written as \\n.\n'
                '- Nothing else may appear inside or after the block: no </script>, no backticks, no other tags, no duplicate </tool_use>.\n'
                '- After the closing </tool_use>, stop generating immediately — no repeated closing tags, no trailing explanation.\n'
                '\nExample of a standard turn:\n'
                'You output: <summary>got dir listing, next read config</summary> (your analysis…) ```<tool_use>{"name": "file_read", "arguments": {"path": "a.txt"}}</tool_use>```\n'
                'The system then returns: <tool_result>file content…</tool_result>, and you continue analyzing or issue the next call.\n'
                f'\n### Tools (mounted, always in effect):\n{tools_json}\n'
            )
        else:
            tool_instruction += (
                '\n格式: ```<tool_use>{"name": "工具名", "arguments": {...}}</tool_use>```\n'
                '输出硬性要求（违反会导致解析失败、任务中断）：\n'
                '- 每个 <tool_use> 块内只放一个完整 JSON 对象；结束前自检：每个 { 都有对应的 }，每个 [ 都有对应的 ]，字符串一律双引号，字符串内的换行必须写成 \\n。\n'
                '- 块内和块后不得出现任何其他内容：不要 </script>，不要反引号，不要其他标签，不要重复 </tool_use>。\n'
                '- 输出闭合的 </tool_use> 后立刻停止生成：不重复闭合标签、不追加任何解释。\n'
                '\n标准回合示例：\n'
                '你的输出: <summary>已获目录列表，下一步读配置文件</summary>（正文分析……）```<tool_use>{"name": "file_read", "arguments": {"path": "a.txt"}}</tool_use>```\n'
                '随后系统返回: <tool_result>文件内容……</tool_result>，你再继续分析或发起下一次调用。\n'
                f'\n### 工具列表 (已挂载，持续有效):\n{tools_json}\n'
            )
        if self.auto_save_tokens and self.last_tools == tools_json:
            tool_instruction = ("\n### Tools: still active, **ready to call**. Protocol unchanged. "
                "Format: <tool_use>{\"name\": \"...\", \"arguments\": {...}}</tool_use> — JSON braces must balance; stop right after </tool_use>; no extra tags/backticks.\n" if _en else
                "\n### 工具库状态：持续有效（code_run/file_read等），**可正常调用**。调用协议沿用。"
                "格式：<tool_use>{\"name\": \"...\", \"arguments\": {...}}</tool_use>，JSON 括号必须配对，</tool_use> 后立即停止，禁止多余标签/反引号。\n")
        else: self.total_cd_tokens = 0
        self.last_tools = tools_json
        return tool_instruction

    def _build_protocol_prompt(self, messages, tools):
        system_content = next((m['content'] for m in messages if m['role'].lower() == 'system'), "")
        history_msgs = [m for m in messages if m['role'].lower() != 'system']
        tool_instruction = self._prepare_tool_instruction(tools)
        system = ""; user = ""
        if system_content: system += f"{system_content}\n"
        system += f"{tool_instruction}"
        for m in history_msgs:
            role = "USER" if m['role'] == 'user' else "ASSISTANT"
            user += f"=== {role} ===\n"
            for tr in m.get('tool_results', []): user += f'<tool_result>{tr["content"]}</tool_result>\n'
            user += str(m['content']) + "\n"
            self.total_cd_tokens += len(user) // 3
        if self.total_cd_tokens > 9000: self.last_tools = ''
        user += "=== ASSISTANT ===\n" 
        return system + user

    def _parse_mixed_response(self, text):
        remaining_text = text; thinking = ''
        think_match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            remaining_text = re.sub(r"<think(?:ing)?>(.*?)</think(?:ing)?>", "", remaining_text, flags=re.DOTALL)
        tool_calls, remaining_text = _parse_text_tool_calls(remaining_text)
        if not tool_calls:
            json_strs = []; errors = []
            if '<tool_use>' in remaining_text:
                weaktoolstr = remaining_text.split('<tool_use>')[-1].strip().strip('><')
                json_str = weaktoolstr if weaktoolstr.endswith('}') else ''
                if json_str == '' and '```' in weaktoolstr and weaktoolstr.split('```')[0].strip().endswith('}'):
                    json_str = weaktoolstr.split('```')[0].strip()
                if json_str: json_strs.append(json_str)
                remaining_text = remaining_text.replace('<tool_use>'+weaktoolstr, "")
            elif '"name":' in remaining_text and '"arguments":' in remaining_text:
                json_match = re.search(r'\{.*"name":.*\}', remaining_text, re.DOTALL)
                if json_match:
                    json_strs.append(json_match.group(0).strip())
                    remaining_text = remaining_text.replace(json_match.group(0), "").strip()
            for json_str in json_strs:
                try:
                    data = tryparse(json_str)
                    func_name = data.get('name') or data.get('function') or data.get('tool')
                    args = data.get('arguments') or data.get('args') or data.get('params') or data.get('parameters')
                    if args is None: args = data
                    if func_name: tool_calls.append(MockToolCall(func_name, args))
                except json.JSONDecodeError:
                    errors.append(f'Failed to parse tool_use JSON: {json_str[:200]}')
                    self.last_tools = ''
                except: pass
            if not tool_calls:
                for e in errors:
                    print(f"[Warn] {e}"); tool_calls.append(MockToolCall('bad_json', {'msg': e}))
        return MockResponse(thinking, remaining_text.strip(), tool_calls, text)

def _scan_json_end(text, start):
    """从 text[start]=='{' 开始做字符串/转义感知的括号平衡扫描，
    返回匹配的 '}' 之后的下标；不平衡（流截断）返回 None。"""
    depth = 0; instr = False; esc = False
    for i in range(start, len(text)):
        c = text[i]
        if instr:
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': instr = False
        elif c == '"': instr = True
        elif c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return i + 1
    return None

def _parse_text_tool_calls(content):
    """Fallback: extract tool calls from text when model doesn't use native tool_use blocks."""
    tcs = []
    # try JSON array: [{"type":"tool_use", "name":..., "input":...}]
    _jp = next((p for p in ['[{"type":"tool_use"', '[{"type": "tool_use"'] if p in content), None)
    if _jp and content.endswith('}]'):
        try:
            idx = content.index(_jp); raw = json.loads(content[idx:])
            tcs = [MockToolCall(b["name"], b.get("input", {}), id=b.get("id", "")) for b in raw if b.get("type") == "tool_use"]
            return tcs, content[:idx].strip()
        except: pass
    # try XML tags: <tool_call>{"name":..., "arguments":...}</tool_call>
    _xp = r"<(?:tool_use|tool_call)>((?:(?!<(?:tool_use|tool_call)>).){15,}?)</(?:tool_use|tool_call)>"
    for s in re.findall(_xp, content, re.DOTALL):
        try:
            d = tryparse(s.strip()); name = d.get('name')
            args = d.get('arguments') or d.get('args') or d.get('input') or {}
            if name: tcs.append(MockToolCall(name, args))
        except: pass
    if tcs: content = re.sub(_xp, "", content, flags=re.DOTALL).strip()
    # qwen3-coder style: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    # (server-side tool-call parser can miss these when the model emits malformed tags,
    #  leaving the call as plain text; without this fallback the agent treats the turn as done)
    _fp = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
    _pp = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
    for _name, _fbody in _fp.findall(content):
        _args = {}
        for _key, _val in _pp.findall(_fbody):
            _key = _key.strip(); _val = _val.strip()
            _stray = '</' + _key + '>'  # model sometimes closes with the key name instead of </parameter>
            if _val.endswith(_stray): _val = _val[:-len(_stray)].rstrip()
            try: _val = json.loads(_val)
            except Exception: pass
            _args[_key] = _val
        tcs.append(MockToolCall(_name.strip(), _args))
    if tcs: content = _fp.sub("", content).strip()
    # DeepSeek DSML style: <｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="tool">
    # <｜｜DSML｜｜ parameter name="k" string="true">v</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>
    # (deepseek 系模型偶发滑回原生 DSML 工具调用格式；不兜底会被当成普通文本 → no_tool → 该轮结束 idle)
    _dsml = r'[|｜]+\s*DSML\s*[|｜]+'
    _dclose = r'</(?:' + _dsml + r'\s*)?'  # 闭合标签有时是裸 </invoke>
    _dip = re.compile(r'<' + _dsml + r'\s*invoke\s+name="([^"]+)"\s*>(.*?)(?:' + _dclose + r'invoke>|$)', re.DOTALL)
    _dpp = re.compile(r'<' + _dsml + r'\s*parameter\s+name="([^"]+)"[^>]*>(.*?)(?:' + _dclose + r'parameter>|$)', re.DOTALL)
    _dsml_found = bool(re.search(_dsml, content))
    for _name, _ibody in _dip.findall(content):
        _args = {}
        for _key, _val in _dpp.findall(_ibody):
            _key = _key.strip(); _val = _val.strip()
            if _val and _val[0] in '[{"':
                try: _val = tryparse(_val)
                except Exception: pass
            _args[_key] = _val
        if set(_args) == {'arguments'}:
            _v = _args['arguments']  # 模型常把 GA 的 arguments 对象整体塞进一个名为 arguments 的参数
            if isinstance(_v, str):
                try: _v = tryparse(_v)
                except Exception: pass
                if isinstance(_v, str):  # 双重编码（JSON 字符串里再套 JSON）
                    try: _v = tryparse(_v)
                    except Exception: pass
            if isinstance(_v, dict): _args = _v
        if len(_args) == 1:
            _v = next(iter(_args.values()))  # 模型把整个 {"name","arguments"} JSON 塞进一个参数
            if isinstance(_v, dict) and isinstance(_v.get('arguments'), dict) and 'name' in _v:
                _args = _v['arguments']
        tcs.append(MockToolCall(_name.strip(), _args))
    # 退化 DSML：模型省略 invoke 行，直接 <parameter name="tool_use">{GA工具JSON}</parameter>，
    # 闭合标签乱序/缺失（如 </invoke></｜｜DSML｜｜ parameter>）。先提取再清理，防止清理把工具 JSON 一起删掉。
    if re.search(r'<(?:[|｜]+\s*DSML\s*[|｜]+\s*)?parameter\s+name="tool_use"', content):
        _tup = re.compile(r'<(?:[|｜]+\s*DSML\s*[|｜]+\s*)?parameter\s+name="tool_use"[^>]*>(\s*\{.*)', re.DOTALL)
        for _m in _tup.finditer(content):
            _raw = _m.group(1)
            _end = re.search(r'</(?:[|｜]+\s*DSML\s*[|｜]+\s*)?parameter>', _raw)
            if _end: _raw = _raw[:_end.start()]
            else:
                _cut = re.search(r'</(?:[|｜]+\s*DSML\s*[|｜]+[^>]*|invoke)\s*>', _raw)
                if _cut: _raw = _raw[:_cut.start()]
            try:
                _d = tryparse(_raw.strip())
                _n = _d.get('name'); _a = _d.get('arguments') or _d.get('args') or _d.get('input') or {}
                if _n: tcs.append(MockToolCall(str(_n).strip(), _a))
            except Exception: pass
    # 混合/畸形格式的通用兜底：上面所有严格格式都没挖到，但文本里还有工具调用标记
    # （deepseek 常把 <tool_use> 和 DSML 混写，如 <tool_use>{...}</｜｜DSML｜｜ parameter>、
    #  <｜｜DSML｜｜ invoke name="x" arguments="{...}"></tool_use>）。不再依赖闭合标签配对，
    # 直接用括号平衡扫描把工具 JSON 挖出来。
    if any(k in content for k in ('<tool_use', '<tool_call', 'DSML', 'invoke name=', '<parameter')):
        _spans = []
        _existing = {(t.function.name, t.function.arguments) for t in tcs}  # 去重：前面严格格式已提取的不再重复添加
        # a) invoke 属性形式：invoke name="X" ... arguments="{...}"（属性值里引号不转义，用平衡扫描取 JSON）
        for _m in re.finditer(r'invoke\s+name="([^"]+)"[^>]*?arguments\s*=\s*"(\s*\{)', content):
            _e = _scan_json_end(content, _m.start(2)) or len(content)
            try:
                _a = tryparse(content[_m.start(2):_e])
                if isinstance(_a, dict):
                    _tc = MockToolCall(_m.group(1).strip(), _a)
                    if (_tc.function.name, _tc.function.arguments) not in _existing:
                        tcs.append(_tc); _spans.append((_m.start(), _e))
            except Exception: pass
        # b) 松散 {"name": "...", "arguments"/"args"/"input": {...}} 形式
        for _m in re.finditer(r'\{\s*"name"\s*:', content):
            if any(s <= _m.start() < e for s, e in _spans): continue
            _e = _scan_json_end(content, _m.start()) or len(content)  # 截断的交给 tryparse 修复
            try: _d = tryparse(content[_m.start():_e])
            except Exception: continue
            if not isinstance(_d, dict): continue
            _n = _d.get('name'); _a = _d.get('arguments') or _d.get('args') or _d.get('input')
            if isinstance(_n, str) and _n.strip() and isinstance(_a, dict):
                _tc = MockToolCall(_n.strip(), _a)
                if (_tc.function.name, _tc.function.arguments) not in _existing:
                    tcs.append(_tc); _spans.append((_m.start(), _e))
        for _s, _e in sorted(_spans, reverse=True):
            content = (content[:_s] + content[_e:]).strip()
        if _spans:
            content = re.sub(r'</?tool_(?:use|call)>', '', content).strip()
    if _dsml_found:
        # 清掉整个 calls 块；无闭合标签时（流截断）只有确实提取到调用才清到末尾，避免误删证据
        _cblk = re.compile(r'<' + _dsml + r'\s*(?:tool_)?calls>(.*?)' + _dclose + r'(?:tool_)?calls>', re.DOTALL)
        if _cblk.search(content):
            content = _cblk.sub('', content).strip()
        elif tcs:
            content = re.sub(r'<' + _dsml + r'\s*(?:tool_)?calls>.*$', '', content, flags=re.DOTALL).strip()
        # 清理残留的孤儿 DSML/invoke/parameter 标签
        content = re.sub(r'</?(?:[|｜]+\s*DSML\s*[|｜]+[^>]*|invoke|parameter[^>]*)\s*>', '', content).strip()
    return tcs, content

def _ensure_text_block(blocks):
    """If response has thinking but no text block, inject a synthetic summary from thinking's first line."""
    if any(b.get("type") == "text" for b in blocks): return None
    th = next((b.get("thinking", "") for b in blocks if b.get("type") == "thinking"), "")
    if not th: return None
    line = th.strip().split('\n', 1)[0]
    txt = "<summary>" + (line[:60] + '...' if len(line) > 60 else line) + "</summary>"
    blocks.insert(1, {"type": "text", "text": txt})
    return txt

def _write_llm_log(label, content, log_path=None, model=''):
    if log_path is False: return
    if not log_path:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'temp/model_responses/model_responses_{os.getpid()}.txt')
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if model: model = f' model={model}'
    with open(log_path, 'a', encoding='utf-8', errors='replace') as f:
        f.write(f"=== {label} === {ts}{model}\n{content}\n\n")

def _json_repair(s, max_iter=40):
    r"""Best-effort repair of near-JSON that json.loads rejects on escapes/tail:
    invalid escapes (\1 \s \w — Python regex fragments in string args), raw control
    chars, model escaping its own closing quote, or truncation mid-string."""
    def _scan(s, end):  # container/string state over s[:end]
        stack, i, instr, esc = [], 0, False, False
        while i < end:
            c = s[i]
            if instr:
                if esc: esc = False
                elif c == '\\': esc = True
                elif c == '"': instr = False
            elif c == '"': instr = True
            elif c in '[{': stack.append(c)
            elif c in ']}':
                if stack and ((c == ']' and stack[-1] == '[') or (c == '}' and stack[-1] == '{')): stack.pop()
                # mismatched closer: leave stack so repair re-closes the still-open container
            i += 1
        return stack, instr, esc
    def _close_open(s):  # close unterminated string + whatever containers are open
        stack, instr, esc = _scan(s, len(s))
        if not stack and not instr and not esc: return None
        return s + ('"' if instr or esc else '') + ''.join(']' if c == '[' else '}' for c in reversed(stack))
    for _ in range(max_iter):
        try: return json.loads(s)
        except json.JSONDecodeError as e:
            if e.pos is None or e.pos > len(s): raise
            if e.msg == 'Invalid \\escape' and s[e.pos] == '\\':
                s = s[:e.pos] + '\\' + s[e.pos:]  # double the stray backslash
            elif e.msg.startswith('Invalid control character') and s[e.pos] in '\n\r\t':
                s = s[:e.pos] + {'\n': '\\n', '\r': '\\r', '\t': '\\t'}[s[e.pos]] + s[e.pos + 1:]
            elif e.msg.startswith('Unterminated string'):
                if s.endswith('\\"'): s = s[:-2] + '"'  # model escaped its own closing quote
                else:  # truncated mid-string: close it plus whatever containers are open
                    s = _close_open(s) or (_ for _ in ()).throw(e)
            elif e.msg == "Expecting ',' delimiter" and e.pos < len(s) and s[e.pos] in ']}':
                # closer for an inner container omitted before an outer one: insert the right closer
                st, _, _ = _scan(s, e.pos)
                if not st: raise
                s = s[:e.pos] + (']' if st[-1] == '[' else '}') + s[e.pos:]
            elif e.pos >= len(s) - 1 and (s2 := _close_open(s)) is not None:
                # structure truncated at end (e.g. missing closing brace): close open containers
                s = s2
            else: raise
    raise ValueError('unrepairable json')

def tryparse(json_str):
    try: return json.loads(json_str)
    except: pass
    json_str = json_str.strip().strip('`').replace('json\n', '', 1).strip()
    try: return json.loads(json_str)
    except: pass
    try: return _json_repair(json_str)  # before truncations: keeps content intact
    except: pass
    try: return json.loads(json_str[:-1])
    except: pass
    if '}' in json_str: json_str = json_str[:json_str.rfind('}') + 1]
    return json.loads(json_str)

class MixinSession:
    """A Session facade backed by multiple routed transport sessions."""
    _TRANSPORT_OVERRIDES = frozenset({
        'stream', 'connect_timeout', 'read_timeout', 'temperature', 'max_tokens',
        'reasoning_effort', 'service_tier', 'thinking_type',
        'thinking_budget_tokens', 'omit_thinking', 'proxies', 'verify',
    })
    _FACADE_STATE = frozenset({'name', 'history', 'system', 'tools', 'lock'})

    def __init__(self, all_sessions, cfg):
        self._retries = cfg.get('max_retries', 3)
        self._base_delay = cfg.get('base_delay', 3.0)
        self._spring_sec = cfg.get('spring_back', 300)
        selected = [all_sessions[i].backend if isinstance(i, int) else
                    next(s.backend for s in all_sessions if type(s) is not dict and s.backend.name == i)
                    for i in cfg.get('llm_nos', [])]
        if not selected: raise ValueError('MixinSession: no sessions selected')
        native_groups = {isinstance(s, NativeClaudeSession) for s in selected}
        if len(native_groups) != 1:
            raise ValueError(f"MixinSession: sessions must be in same group (Native or non-Native), got {[type(s).__name__ for s in selected]}")

        self._sessions = [copy.copy(s) for s in selected]
        for s in self._sessions: s.max_retries = 0
        self._native = native_groups.pop()
        self._ask_impl = selected[0].ask.__func__
        self._cur_idx, self._switched_at = 0, 0.0

        primary = self._sessions[0]
        self.name = '|'.join(s.name for s in self._sessions)
        self.history = copy.deepcopy(primary.history)
        self.system = primary.system
        self.tools = getattr(primary, 'tools', None)
        self.lock = threading.Lock()
    @property
    def primary(self): return self._sessions[0]
    @property
    def current(self): return self._sessions[self._cur_idx]
    @property
    def current_name(self): return self.current.name
    def __getattr__(self, name): return getattr(self.current, name)
    def __setattr__(self, name, value):
        sessions = self.__dict__.get('_sessions')
        if sessions and name in self._TRANSPORT_OVERRIDES:
            for s in sessions: setattr(s, name, value)
            return
        node_owns = sessions and any(
            name in s.__dict__ or any(name in cls.__dict__ for cls in type(s).__mro__)
            for s in sessions)
        if node_owns and name not in self._FACADE_STATE: raise AttributeError(f"MixinSession.{name} is node-specific and read-only")
        object.__setattr__(self, name, value)
    def ask(self, prompt):
        self._pick()  # Select the node before ask() reads its context limits.
        return self._ask_impl(self, prompt)
    def make_messages(self, messages): return messages
    def _pick(self):
        if self._cur_idx and time.time() - self._switched_at > self._spring_sec: self._cur_idx = 0
        return self._cur_idx
    def _prepare(self, idx, messages):
        session = self._sessions[idx]
        session.system = self.system
        session.tools = openai_tools_to_claude(self.tools) if self.tools and type(session) is NativeClaudeSession else self.tools
        return messages if self._native else session.make_messages(messages)
    def raw_ask(self, messages):
        base, n = self._pick(), len(self._sessions)
        test_error = lambda x: isinstance(x, str) and x.lstrip().startswith(('!!!Error:', '[Error:'))
        for attempt in range(self._retries + 1):
            idx = (base + attempt) % n
            session = self._sessions[idx]
            gen = session.raw_ask(self._prepare(idx, messages))
            print(f'[MixinSession] Using session ({session.name})')
            last_chunk, return_val, yielded = None, [], False
            try:
                while True:
                    chunk = next(gen); last_chunk = chunk
                    if not yielded and test_error(chunk): continue
                    yield chunk; yielded = True
            except StopIteration as e: return_val = e.value or []
            is_err = test_error(last_chunk)
            if not is_err:
                if attempt > 0: self._cur_idx = idx; self._switched_at = time.time()
                elif isinstance(last_chunk, str) and '[!!! 流异常中断' in last_chunk and n > 1:
                    self._cur_idx = (idx + 1) % n; self._switched_at = time.time()
                    print(f'[MixinSession] Partial failure, next call → s{self._cur_idx} ({self.current.name})')
                return return_val
            if attempt >= self._retries:
                yield last_chunk; return return_val
            nxt = (base + attempt + 1) % n
            if nxt == base:
                rnd = (attempt + 1) // n
                delay = min(30, self._base_delay * (1.5 ** rnd))
                print(f'[MixinSession] {last_chunk[:80]}, round {rnd} exhausted, retry in {delay:.1f}s')
                time.sleep(delay)
            else:
                print(f'[MixinSession] {last_chunk[:80]}, retry {attempt+1}/{self._retries} (s{idx}→s{nxt})')

THINKING_PROMPT_ZH = """
### 行动规范（持续有效）
每次回复（含工具调用轮）都先在回复文字中包含一个<summary></summary> 中输出极简单行（<30字）物理快照：上次结果新信息+本次意图。此内容进入长期工作记忆。
\n**若用户需求未完成，必须进行工具调用！**
""".strip()
THINKING_PROMPT_EN = """
### Action Protocol (always in effect)
The reply body should first include a minimal one-line (<30 words) physical snapshot in <summary></summary>: new info from last result + current intent. This goes into long-term working memory.
\n**If the user's request is not yet complete, tool calls are required!**
""".strip()

class NativeToolClient:
    @staticmethod
    def _thinking_prompt(): return THINKING_PROMPT_EN if os.environ.get('GA_LANG') == 'en' else THINKING_PROMPT_ZH
    def __init__(self, backend):
        self.backend = backend
        self.backend.system = self._thinking_prompt()
        self.name = self.backend.name
        self._pending_tool_ids = []
        self.log_path = None
    def set_system(self, extra_system):
        combined = f"{extra_system}\n\n{self._thinking_prompt()}" if extra_system else self._thinking_prompt()
        if combined != self.backend.system: print(f"[Debug] Updated system prompt, length {len(combined)} chars.")
        self.backend.system = combined
    def chat(self, messages, tools=None):
        if tools: self.backend.tools = tools
        if not self.backend.history: self._pending_tool_ids = []
        combined_content = []; resp = None; tool_results = []
        for msg in messages:
            c = msg.get('content', '')
            if msg['role'] == 'system': 
                self.set_system(c); continue
            if isinstance(c, str): combined_content.append({"type": "text", "text": c})
            elif isinstance(c, list): combined_content.extend(c)
            if msg['role'] == 'user' and msg.get('tool_results'): tool_results.extend(msg['tool_results'])
        tr_id_set = set();  tool_result_blocks = []
        for tr in tool_results:
            tool_use_id, content = tr.get("tool_use_id", ""), tr.get("content", "")
            tr_id_set.add(tool_use_id)
            if tool_use_id: tool_result_blocks.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": tr.get("content", "")})
            else: combined_content = [{"type": "text", "text": f'<tool_result>{content}</tool_result>'}] + combined_content
        for tid in self._pending_tool_ids:
            if tid not in tr_id_set: tool_result_blocks.append({"type": "tool_result", "tool_use_id": tid, "content": ""})
        self._pending_tool_ids = []
        # Filter whitespace-only text blocks that cause 400 on strict API proxies
        filtered_content = [c for c in combined_content if c.get("text", "").strip()]
        final_content = tool_result_blocks + filtered_content
        if not final_content: final_content = [{"type": "text", "text": "."}]
        merged = {"role": "user", "content": final_content}
        prompt_raw = '{"role": "user", "content": [\n' + ",\n".join(json.dumps(b, ensure_ascii=False) for b in final_content) + "]}"
        _write_llm_log('Prompt', prompt_raw, self.log_path)
        gen = self.backend.ask(merged)
        try:
            while True: 
                chunk = next(gen); yield chunk
        except StopIteration as e: resp = e.value
        if resp: _write_llm_log('Response', resp.raw, self.log_path, model=self.backend.model)
        if resp and hasattr(resp, 'tool_calls') and resp.tool_calls: self._pending_tool_ids = [tc.id for tc in resp.tool_calls]
        return resp

def resolve_session(cfg_name):
    cfg = reload_mykeys()[0].get(cfg_name)
    if not cfg: raise ValueError(f"Config '{cfg_name}' not in mykey")
    cfg['_mykey_name'] = cfg_name
    if 'native' in cfg_name: return (NativeClaudeSession if 'claude' in cfg_name else NativeOAISession)(cfg=cfg)
    if 'claude' in cfg_name: return ClaudeSession(cfg=cfg)
    return LLMSession(cfg=cfg) if 'oai' in cfg_name else None

def resolve_client(cfg_name):
    s = resolve_session(cfg_name)
    return (NativeToolClient(s) if isinstance(s, (NativeClaudeSession, NativeOAISession)) else ToolClient(s)) if s else None

def fast_ask(prompt, cfg_name):
    sess = resolve_session(cfg_name)
    if not sess: raise ValueError(f"fast_ask: '{cfg_name}' unsupported")
    return "".join(sess.raw_ask([{"role": "user", "content": prompt}]))
