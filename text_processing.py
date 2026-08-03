# text_processing.py
import re
from typing import Dict, Any, Tuple, List, Iterator, Union
from logging_utils import _LogWrapper

# ─── Formatting Marker Stripping ──────────────────────────────────────────
_FORMATTING_MARKER = "Formatting re-enabled"
_THINK_TAG_SPLIT_RE = re.compile(r'(</?think>|</?thought>)')
_FENCE_RE = re.compile(r"^(?P<fence>[`~]{3,})(?P<lang>.*)$")
_THOUGHT_BLOCK_RE = re.compile(r'<(thought|think)>.*?</\1>', flags=re.DOTALL)

def _strip_formatting_marker(text: str) -> str:
    """
    Remove the leading 'Formatting re-enabled' marker (if present) and
    trim one optional newline / carriage-return / space that follows it.
    """
    if text.startswith(_FORMATTING_MARKER):
        return text[len(_FORMATTING_MARKER):].lstrip("\r\n ")
    return text

# ─── Contextual Replacement Helper ──────────────────────────────────────────
def _apply_contextual_replacements(text: str, replacements: Dict, log_wrapper) -> str:
    """
    Apply contextual replacements based on position rules.

    Replacements can be:
    - Simple: {'old': 'new'} - replaces anywhere (backward compatibility)
    - Contextual: {'old': {'replacement': 'new', 'context': 'start|end|middle|anywhere|outside_code'}}
    """
    if not replacements:
        return text

    lines = text.split('\n')
    processed_lines = []

    in_code_block = False

    for line in lines:
        processed_line = line

        if '```' in line:
            fence_count = line.count('```')
            if fence_count % 2 == 1:
                in_code_block = not in_code_block

        for old, replacement_config in replacements.items():
            if isinstance(replacement_config, str):
                new = replacement_config
                context = "anywhere"
            else:
                new = replacement_config.get('replacement', replacement_config)
                context = replacement_config.get('context', 'anywhere')

            if in_code_block:
                continue

            if context == "anywhere":
                processed_line = processed_line.replace(old, new)
            elif context == "start":
                if processed_line.startswith(old):
                    processed_line = new + processed_line[len(old):]
            elif context == "end":
                if processed_line.endswith(old):
                    processed_line = processed_line[:-len(old)] + new
            elif context == "middle":
                if old in processed_line and not processed_line.startswith(old) and not processed_line.endswith(old):
                    start_idx = 0
                    while True:
                        idx = processed_line.find(old, start_idx)
                        if idx == -1:
                            break
                        if idx == 0 or idx + len(old) == len(processed_line):
                            start_idx = idx + len(old)
                            continue
                        processed_line = processed_line[:idx] + new + processed_line[idx + len(old):]
                        start_idx = idx + len(new)
            elif context == "outside_code":
                processed_line = processed_line.replace(old, new)

        processed_lines.append(processed_line)

    return '\n'.join(processed_lines)

def _has_contextual_replacement_rules(app_config: Dict[str, Any]) -> bool:
    """Check if the configuration contains contextual replacement rules that need smart buffering."""
    if not app_config.get('REPLACE_CHARACTERS_ENABLED', False):
        return False

    replacements = app_config.get('REPLACEMENTS', {})
    for pattern, replacement_config in replacements.items():
        if isinstance(replacement_config, dict):
            context = replacement_config.get('context', 'anywhere')
            if context in ['start', 'end', 'middle', 'outside_code']:
                return True
        elif isinstance(replacement_config, str) and len(pattern) > 1:
            return True

    return False

# ─── Non-Streaming Text Processing Helper ────────────────────────────────────
_STRIP_CHARS_REGEX_CACHE: Dict[Tuple[str, ...], re.Pattern] = {}
def _process_non_stream_text_content(text_content: str, app_config: Dict[str, Any], logger=None, debug_status=False, model_id=None) -> str:
    log_wrapper = _LogWrapper(logger, debug_status, "NonStreamTextProc")

    if not isinstance(text_content, str):
        if logger and debug_status:
            log_wrapper.warning(f"Input to _process_non_stream_text_content is not a string (type: {type(text_content)}). Returning as is.")
        return text_content

    original_len_for_log = len(text_content)

    processed_text = text_content

    if app_config.get('STRIP_CHARACTERS_ENABLED', False):
        chars_to_strip = app_config.get('CHARACTERS_TO_STRIP', [])
        if chars_to_strip:
            log_wrapper.debug(f"Applying strip characters: {chars_to_strip}")
            key = tuple(chars_to_strip)
            compiled = _STRIP_CHARS_REGEX_CACHE.get(key)
            if compiled is None:
                strip_pattern = f'[{re.escape(" ".join(chars_to_strip))}]'
                compiled = re.compile(strip_pattern)
                _STRIP_CHARS_REGEX_CACHE[key] = compiled
            processed_text = compiled.sub('', processed_text)

    if app_config.get('REPLACE_CHARACTERS_ENABLED', False):
        replacements = app_config.get('REPLACEMENTS', {})
        if replacements:
            log_wrapper.debug(f"Applying replacements: {replacements}")
        processed_text = _apply_contextual_replacements(processed_text, replacements, log_wrapper)

    strip_config = app_config.get('STRIP_THOUGHT_BLOCKS', False)
    should_strip = False
    if isinstance(strip_config, bool):
        should_strip = strip_config
    elif isinstance(strip_config, list) and model_id:
        should_strip = model_id in strip_config

    if should_strip:
        log_wrapper.debug("Stripping thought/think blocks from non-stream response.")
        processed_text = _THOUGHT_BLOCK_RE.sub('', processed_text)

    if len(processed_text) != original_len_for_log:
        log_wrapper.debug(f"Text processing applied. Original len: {original_len_for_log}, New len: {len(processed_text)}")

    return processed_text

# ─── System Prompt Stripping ─────────────────────────────────────────────
_REGEX_CACHE: Dict[Tuple[str, ...], List[re.Pattern]] = {}
def _strip_dynamic_system_prompt_content(text_content: str, app_config: Dict[str, Any], logger=None, debug_status=False) -> str:
    """
    Removes dynamically generated lines from a system prompt using regex patterns
    defined in the application configuration.
    """
    log = _LogWrapper(logger, debug_status, "SystemPromptStrip")

    if not app_config.get('DYNAMIC_SYSTEM_PROMPT_STRIP_ENABLED', False):
        return text_content

    if not isinstance(text_content, str):
        log.warning(f"Input is not a string (type: {type(text_content)}). Returning as is.")
        return text_content

    original_len = len(text_content)
    patterns_to_strip_str = app_config.get('DYNAMIC_SYSTEM_PROMPT_PATTERNS_TO_STRIP', [])

    if not patterns_to_strip_str:
        return text_content

    cache_key = tuple(patterns_to_strip_str)
    compiled_patterns = _REGEX_CACHE.get(cache_key)
    if compiled_patterns is None:
        compiled_patterns = []
        for p_str in patterns_to_strip_str:
            try:
                compiled_patterns.append(re.compile(p_str, re.MULTILINE))
            except re.error as e:
                log.error(f"Invalid regex pattern in config: '{p_str}'. Error: {e}")
                continue
        _REGEX_CACHE[cache_key] = compiled_patterns

    processed_text = text_content
    for pattern in compiled_patterns:
        processed_text = pattern.sub("", processed_text)

    if len(processed_text) != original_len:
        log.debug(f"Dynamic system prompt content stripped. Original len: {original_len}, New len: {len(processed_text)}")

    return processed_text

# ─── Smart Buffering Context-Aware Streaming Text Processor ─────────────────
class ContextualRule:
    """Represents a contextual replacement rule with buffering logic."""
    def __init__(self, pattern: str, replacement: str, context: str):
        self.pattern = pattern
        self.replacement = replacement
        self.context = context
        self.pattern_len = len(pattern)

    def needs_buffering(self) -> bool:
        return self.context in ['start', 'end', 'middle', 'outside_code']

    def get_buffer_size(self) -> int:
        if self.context == 'end':
            return self.pattern_len + 2
        elif self.context == 'start':
            return self.pattern_len + 1
        elif self.context == 'middle':
            return self.pattern_len + 2
        return self.pattern_len

class SmartBuffer:
    """Manages buffering for contextual rule evaluation."""
    def __init__(self, max_size: int = 20):
        self.buffer = ""
        self.max_size = max_size
        self.line_start_pos = 0
        self.is_true_line_start = True

    def add_char(self, char: str) -> None:
        self.buffer += char
        if char == '\n':
            self.line_start_pos = len(self.buffer)
            self.is_true_line_start = True

        if len(self.buffer) > self.max_size:
            trim_amount = len(self.buffer) - self.max_size
            if trim_amount < self.line_start_pos:
                self.buffer = self.buffer[trim_amount:]
                self.line_start_pos -= trim_amount
            else:
                self.buffer = self.buffer[self.line_start_pos:]
                self.line_start_pos = 0
                self.is_true_line_start = False

    def get_current_line(self) -> str:
        return self.buffer[self.line_start_pos:].rstrip('\n')

    def ends_with_pattern(self, pattern: str) -> bool:
        return self.buffer.endswith(pattern)

    def pattern_at_line_start(self, pattern: str) -> bool:
        if not self.is_true_line_start:
            return False
        current_line = self.get_current_line()
        return current_line.startswith(pattern)

    def pattern_in_middle(self, pattern: str) -> bool:
        current_line = self.get_current_line()
        if pattern not in current_line:
            return False
        return not current_line.startswith(pattern) and not current_line.endswith(pattern)

    def clear(self) -> str:
        content = self.buffer
        self.buffer = ""
        self.line_start_pos = 0
        self.is_true_line_start = True
        return content

class ContextAwareStreamingProcessor:
    """Enhanced streaming processor with smart buffering for contextual rules."""

    def __init__(self, app_config: Dict[str, Any], logger, debug_status: bool, log_prefix: str = "CONTEXT_PROC"):
        self.config = app_config
        self.log = _LogWrapper(logger, debug_status, log_prefix)

        self.contextual_rules = []
        self.simple_char_replacements = {}

        if app_config.get('REPLACE_CHARACTERS_ENABLED', False):
            replacements = app_config.get('REPLACEMENTS', {})
            for pattern, replacement_config in replacements.items():
                if isinstance(replacement_config, str):
                    if len(pattern) == 1:
                        self.simple_char_replacements[pattern] = replacement_config
                    else:
                        rule = ContextualRule(pattern, replacement_config, 'anywhere')
                        self.contextual_rules.append(rule)
                else:
                    replacement = replacement_config.get('replacement', '')
                    context = replacement_config.get('context', 'anywhere')
                    rule = ContextualRule(pattern, replacement, context)
                    self.contextual_rules.append(rule)

        self.smart_buffer = SmartBuffer()
        self.in_code_block = False
        self.pending_output = ""
        self.is_start_of_line = True

        self.strip_enabled = bool(app_config.get('STRIP_CHARACTERS_ENABLED', False))
        raw_chars_to_strip = app_config.get('CHARACTERS_TO_STRIP', [])
        if isinstance(raw_chars_to_strip, (list, tuple, set)):
            self.chars_to_strip = frozenset(raw_chars_to_strip)
        else:
            self.chars_to_strip = frozenset()

        self.log.debug(f"Initialized with {len(self.contextual_rules)} contextual rules, {len(self.simple_char_replacements)} simple replacements")
        self.log.debug(f"  Strip enabled: {self.strip_enabled}, Chars to strip: {list(self.chars_to_strip)}")
        for rule in self.contextual_rules:
            self.log.debug(f"  Rule: '{rule.pattern}' -> '{rule.replacement}' (context: {rule.context})")

    def _detect_code_block_boundary(self, char: str) -> None:
        if char == '`':
            buffer_content = self.smart_buffer.buffer + char
            lines = buffer_content.split('\n')
            current_line = lines[-1] if lines else ""
            stripped_line = current_line.strip()

            backtick_count = 0
            for c in stripped_line:
                if c == '`':
                    backtick_count += 1
                else:
                    break

            if backtick_count == 3 and stripped_line.startswith('```'):
                self.in_code_block = not self.in_code_block
                self.log.debug(f"Code fence detected: '{stripped_line}' - in_code_block={self.in_code_block}")

    def _apply_simple_replacements(self, char: str) -> str:
        if self.strip_enabled and char in self.chars_to_strip:
            return ''

        return self.simple_char_replacements.get(char, char)

    def _check_contextual_rules(self) -> str:
        output = ""

        for rule in self.contextual_rules:
            if self.in_code_block:
                self.log.debug(f"Skipping rule '{rule.pattern}' - inside code block")
                continue

            if not rule.needs_buffering():
                if rule.context == 'anywhere' and self.smart_buffer.ends_with_pattern(rule.pattern):
                    buffer_content = self.smart_buffer.buffer
                    if buffer_content.endswith(rule.pattern):
                        self.smart_buffer.buffer = buffer_content[:-len(rule.pattern)]
                        output += rule.replacement
                        self.log.debug(f"Applied 'anywhere' rule: '{rule.pattern}' -> '{rule.replacement}'")
                        break
                continue

            if len(self.smart_buffer.buffer) < rule.get_buffer_size():
                continue

            if rule.context == 'end':
                current_line = self.smart_buffer.get_current_line()
                if current_line.endswith(rule.pattern) and (
                    self.smart_buffer.buffer.endswith(rule.pattern + '\n') or
                    (len(self.smart_buffer.buffer) >= rule.get_buffer_size() and
                     self.smart_buffer.buffer.endswith(rule.pattern))
                ):
                    buffer_content = self.smart_buffer.buffer
                    if buffer_content.endswith(rule.pattern + '\n'):
                        self.smart_buffer.buffer = buffer_content[:-len(rule.pattern)-1]
                        output += rule.replacement + '\n'
                    elif buffer_content.endswith(rule.pattern):
                        self.smart_buffer.buffer = buffer_content[:-len(rule.pattern)]
                        output += rule.replacement
                    self.log.debug(f"Applied 'end' rule: '{rule.pattern}' -> '{rule.replacement}'")
                    break

            elif rule.context == 'outside_code':
                if rule.pattern in self.smart_buffer.buffer:
                    buffer_content = self.smart_buffer.buffer
                    new_content = buffer_content.replace(rule.pattern, rule.replacement)
                    self.smart_buffer.buffer = new_content
                    if self.smart_buffer.line_start_pos > len(new_content):
                        self.smart_buffer.line_start_pos = len(new_content)
                    self.log.debug(f"Applied 'outside_code' rule: '{rule.pattern}' -> '{rule.replacement}'")
                    break

            elif rule.context == 'start':
                if self.smart_buffer.pattern_at_line_start(rule.pattern):
                    current_line = self.smart_buffer.get_current_line()
                    if current_line.startswith(rule.pattern):
                        new_line = rule.replacement + current_line[len(rule.pattern):]
                        buffer_before_line = self.smart_buffer.buffer[:self.smart_buffer.line_start_pos]
                        line_ending = '\n' if self.smart_buffer.buffer.endswith('\n') else ''
                        self.smart_buffer.buffer = buffer_before_line + new_line + line_ending
                        self.log.debug(f"Applied 'start' rule: '{rule.pattern}' -> '{rule.replacement}'")
                        break

            elif rule.context == 'middle':
                if self.smart_buffer.pattern_in_middle(rule.pattern):
                    current_line = self.smart_buffer.get_current_line()
                    new_line = current_line.replace(rule.pattern, rule.replacement)
                    buffer_before_line = self.smart_buffer.buffer[:self.smart_buffer.line_start_pos]
                    line_ending = '\n' if self.smart_buffer.buffer.endswith('\n') else ''
                    self.smart_buffer.buffer = buffer_before_line + new_line + line_ending
                    self.log.debug(f"Applied 'middle' rule: '{rule.pattern}' -> '{rule.replacement}'")
                    break

        return output

    def _should_flush_buffer(self, new_char: str) -> bool:
        buffer_len = len(self.smart_buffer.buffer)

        if new_char == '\n':
            return True

        if not self.in_code_block and buffer_len > 0:
            last_char = self.smart_buffer.buffer[-1]
            if last_char in (' ', '\t'):
                return False

        if buffer_len > 20:
            return True

        max_pattern_len = max((len(rule.pattern) for rule in self.contextual_rules), default=0)
        if buffer_len >= max_pattern_len + 2:
            return True

        return False

    def process_chunk(self, text_chunk: str) -> Iterator[str]:
        if not text_chunk:
            return

        # Split by thinking tags to ensure they are yielded instantly and isolated
        parts = _THINK_TAG_SPLIT_RE.split(text_chunk)
        for part in parts:
            if not part:
                continue

            if part in ['<think>', '</think>', '<thought>', '</thought>']:
                # Flush existing buffer and contextual rules first
                rule_output = self._check_contextual_rules()
                if rule_output:
                    yield rule_output

                remaining = self.smart_buffer.clear()
                if remaining:
                    yield remaining

                # Yield the tag itself
                yield part
                self.log.debug(f"Flushed buffer and yielded isolated tag: {part}")
            else:
                for char in part:
                    processed_char = self._apply_simple_replacements(char)
                    if not processed_char:
                        continue

                    self._detect_code_block_boundary(processed_char)

                    if processed_char == '\n' and not self.in_code_block:
                        buf = self.smart_buffer.buffer
                        start = self.smart_buffer.line_start_pos
                        end = len(buf)
                        trim_idx = end - 1
                        while trim_idx >= start and trim_idx < len(buf) and buf[trim_idx] in (' ', '\t'):
                            trim_idx -= 1
                        if trim_idx < end - 1:
                            self.smart_buffer.buffer = buf[:trim_idx+1]
                            self.log.debug("Stripped trailing horizontal whitespace before newline outside code block")

                    self.smart_buffer.add_char(processed_char)

                    # --- Tag Detection (Isolated Yielding, fallback for split tags) ---
                    detected_tag = None
                    for tag in ['<think>', '</think>', '<thought>', '</thought>']:
                        if self.smart_buffer.buffer.endswith(tag):
                            detected_tag = tag
                            break

                    if detected_tag:
                        content_before = self.smart_buffer.buffer[:-len(detected_tag)]
                        if content_before:
                            yield content_before
                        yield detected_tag
                        self.smart_buffer.clear()
                        self.log.debug(f"Flushed buffer and yielded {detected_tag} tag (in-loop).")
                        continue

                    if processed_char == '\n':
                        self.is_start_of_line = True
                    elif not processed_char.isspace():
                        self.is_start_of_line = False

                    if self._should_flush_buffer(processed_char):
                        rule_output = self._check_contextual_rules()
                        if rule_output:
                            yield rule_output

                        buffer_len = len(self.smart_buffer.buffer)
                        if buffer_len > 8:
                            max_pattern_len = max((len(rule.pattern) for rule in self.contextual_rules), default=2)
                            keep_size = max(max_pattern_len + 1, 3)

                            if buffer_len > keep_size:
                                safe_output_len = buffer_len - keep_size
                                safe_output = self.smart_buffer.buffer[:safe_output_len]
                                self.smart_buffer.buffer = self.smart_buffer.buffer[safe_output_len:]

                                if self.smart_buffer.line_start_pos >= safe_output_len:
                                    self.smart_buffer.line_start_pos -= safe_output_len
                                else:
                                    self.smart_buffer.line_start_pos = 0
                                    self.smart_buffer.is_true_line_start = False

                                if safe_output:
                                    yield safe_output

        rule_output = self._check_contextual_rules()
        if rule_output:
            yield rule_output

        buffer_len = len(self.smart_buffer.buffer)
        if buffer_len > 4:
            max_pattern_len = max((len(rule.pattern) for rule in self.contextual_rules), default=2)
            keep_size = max(max_pattern_len + 1, 3)
            if buffer_len > keep_size:
                safe_output_len = buffer_len - keep_size
                safe_output = self.smart_buffer.buffer[:safe_output_len]
                self.smart_buffer.buffer = self.smart_buffer.buffer[safe_output_len:]
            else:
                safe_output = ""
            if self.smart_buffer.line_start_pos > len(self.smart_buffer.buffer):
                self.smart_buffer.line_start_pos = 0
            if safe_output:
                yield safe_output

    def finalize(self) -> Iterator[str]:
        if self.smart_buffer.buffer:
            rule_output = self._check_contextual_rules()
            if rule_output:
                yield rule_output

            remaining = self.smart_buffer.clear()
            if remaining:
                yield remaining

# ─── Core Text Processing (Streaming) ──────────────────────────────────────
class StreamingTextProcessor:
    def __init__(self, app_config: Dict[str, Any], logger, debug_status: bool, log_prefix: str = "TEXT_PROC"):
        self.config = app_config
        self.log = _LogWrapper(logger, debug_status, log_prefix)

        self.collapse_spaces = self.config.get('COLLAPSE_DOUBLE_SPACES_OUTSIDE_CODE', True)
        self.sanitize_dollars = self.config.get('SANITIZE_DOLLAR_SIGNS', False)
        self.concise_mode = self.config.get('CONCISEMODE', False)
        self.bullet_replacement = self.config.get('BULLET_REPLACEMENT_CHAR', '→')

        self.strip_enabled = bool(self.config.get('STRIP_CHARACTERS_ENABLED', False))
        raw_chars_to_strip = self.config.get('CHARACTERS_TO_STRIP', [])
        if not isinstance(raw_chars_to_strip, (list, tuple, set)):
            self.log.warning(f"CHARACTERS_TO_STRIP config is not a list, tuple, or set (type: {type(raw_chars_to_strip)}). Treating as empty.")
            self.chars_to_strip = frozenset()
        else:
            self.chars_to_strip = frozenset(raw_chars_to_strip)

        self.replacements_enabled = bool(self.config.get('REPLACE_CHARACTERS_ENABLED', False))
        self.replacements = self.config.get('REPLACEMENTS', {})

        self.log.debug(f"Initialized StreamingTextProcessor. Strip enabled: {self.strip_enabled}, Chars to strip: {list(self.chars_to_strip)}")
        self.log.debug(f"  Replacements enabled: {self.replacements_enabled}, Replacements: {self.replacements}")

        self.in_code_block = False
        self._current_fence_type: Optional[str] = None
        self.last_char_yielded_category = None
        self.is_start_of_line = True
        self.is_first_chunk = True

    def _transform_special_list_item_line(self, line_content: str) -> str:
        if line_content.startswith('• ') and line_content.endswith('  ') and len(line_content) > 3:
            transformed = '- ' + line_content[2:-2]
            self.log.debug(
                f"Transformed list item line ({len(line_content)} -> {len(transformed)} chars)."
            )
            return transformed
        return line_content

    def _process_char_stateless_enhanced(self, char: str) -> str:
        if self.strip_enabled and char in self.chars_to_strip:
            return ''

        if self.replacements_enabled:
            if char in self.replacements:
                replacement_config = self.replacements[char]
                if isinstance(replacement_config, str):
                    return replacement_config
                else:
                    context = replacement_config.get('context', 'anywhere')
                    if context == 'anywhere':
                        return replacement_config.get('replacement', char)
                    return char

        return char

    def process_chunk(self, text_chunk: str) -> Iterator[str]:
        if self.is_first_chunk and not self.in_code_block:
            original_len = len(text_chunk)
            text_chunk = text_chunk.lstrip()
            if len(text_chunk) != original_len:
                self.log.debug(f"Stripped leading whitespace from initial chunk. Original len: {original_len}, New len: {len(text_chunk)}")

            if text_chunk:
                self.is_first_chunk = False

        if not text_chunk:
            return

        # Split by thinking tags to ensure they are yielded instantly and isolated
        parts = re.split(r'(</?think>|</?thought>)', text_chunk)
        for part in parts:
            if not part:
                continue

            if part in ['<think>', '</think>', '<thought>', '</thought>']:
                yield part
            else:
                lines = part.splitlines(True)

                for line_idx, line_with_ending in enumerate(lines):
                    line_content: str
                    line_ending: str = ""

                    if line_with_ending.endswith("\r\n"):
                        line_content = line_with_ending[:-2]
                        line_ending = "\r\n"
                    elif line_with_ending.endswith("\n"):
                        line_content = line_with_ending[:-1]
                        line_ending = "\n"
                    else:
                        line_content = line_with_ending

                    stripped_line_for_fence_check = line_content.strip()
                    fence_match = _FENCE_RE.match(stripped_line_for_fence_check)
                    is_fence_line = False

                    if fence_match:
                        fence_chars = fence_match.group("fence")
                        lang_specifier = fence_match.group("lang").strip()

                        if not self.in_code_block:
                            self.in_code_block = True
                            self._current_fence_type = fence_chars
                            self.log.debug(f"Entered code block with fence: {fence_chars} {lang_specifier}")
                            yield fence_chars + ((" " + lang_specifier) if lang_specifier else "")
                            if line_ending: yield line_ending
                            is_fence_line = True
                        elif self.in_code_block and fence_chars == self._current_fence_type and not lang_specifier:
                            self.in_code_block = False
                            self._current_fence_type = None
                            self.log.debug(f"Exited code block with fence: {fence_chars}")
                            yield fence_chars
                            if line_ending: yield line_ending
                            is_fence_line = True

                    if is_fence_line:
                        if line_ending:
                            self.is_start_of_line = True
                            self.last_char_yielded_category = 'newline'
                        else:
                            self.is_start_of_line = False
                            self.last_char_yielded_category = 'char'
                        continue

                    if self.in_code_block:
                        yield line_with_ending
                        if line_ending:
                            self.is_start_of_line = True
                            self.last_char_yielded_category = 'newline'
                        else:
                            self.is_start_of_line = False
                            if line_content:
                                last_char = line_content[-1]
                                self.last_char_yielded_category = 'space' if (last_char.isspace() and last_char != '\n') else 'char'
                        continue

                    transformed_line_content = self._transform_special_list_item_line(line_content)
                    current_line_output_chars = []
                    is_at_start_of_current_line_processing = True
                    last_char_in_buffer_category = None

                    for original_char_idx, original_char in enumerate(transformed_line_content):
                        chars_to_process_segment = self._process_char_stateless_enhanced(original_char)

                        for char_to_process in chars_to_process_segment:
                            segment_for_this_processed_char = []
                            is_bullet_transformed_this_iteration = False

                            if is_at_start_of_current_line_processing:
                                if char_to_process == '•':
                                    segment_for_this_processed_char.append(self.bullet_replacement)
                                    is_bullet_transformed_this_iteration = True
                                if not (char_to_process.isspace() and char_to_process != '\n'):
                                    is_at_start_of_current_line_processing = False

                            if not is_bullet_transformed_this_iteration:
                                is_current_char_horizontal_space = char_to_process.isspace() and char_to_process != '\n'

                                should_collapse = (
                                    self.collapse_spaces and
                                    is_current_char_horizontal_space and
                                    last_char_in_buffer_category == 'space' and
                                    not is_at_start_of_current_line_processing
                                )

                                if self.collapse_spaces and is_current_char_horizontal_space:
                                    is_first_space_after_char = False
                                    if current_line_output_chars:
                                        if last_char_in_buffer_category == 'char':
                                            is_first_space_after_char = True
                                    else:
                                        if self.last_char_yielded_category == 'char':
                                            is_first_space_after_char = True

                                    is_true_indentation_space = self.is_start_of_line and is_at_start_of_current_line_processing

                                    if is_true_indentation_space or is_first_space_after_char:
                                        segment_for_this_processed_char.append(char_to_process)
                                        last_char_in_buffer_category = 'space'
                                    else:
                                        pass
                                else:
                                    segment_for_this_processed_char.append(char_to_process)
                                    if is_current_char_horizontal_space:
                                        last_char_in_buffer_category = 'space'
                                    else:
                                        last_char_in_buffer_category = 'char'

                            if segment_for_this_processed_char:
                                current_line_output_chars.extend(segment_for_this_processed_char)

                    if current_line_output_chars:
                        line_to_yield_str = "".join(current_line_output_chars)

                        if self.collapse_spaces and line_ending:
                            original_len_for_log = len(line_to_yield_str)
                            line_to_yield_str = line_to_yield_str.rstrip(' \t')
                            if len(line_to_yield_str) != original_len_for_log:
                                self.log.debug(
                                    "Rstripped trailing spaces before newline "
                                    f"({original_len_for_log} -> {len(line_to_yield_str)} chars)."
                                )

                        if line_to_yield_str:
                            yield line_to_yield_str
                            self.is_start_of_line = False
                            last_char_of_yielded_line = line_to_yield_str[-1]
                            if last_char_of_yielded_line.isspace() and last_char_of_yielded_line != '\n':
                                self.last_char_yielded_category = 'space'
                            else:
                                self.last_char_yielded_category = 'char'

                    if line_ending:
                        yield line_ending
                        self.is_start_of_line = True
                        self.last_char_yielded_category = 'newline'
