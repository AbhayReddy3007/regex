streamlit.runtime.caching.cache_errors.UnhashableParamError: Cannot hash argument 'model' (of type `google.generativeai.generative_models.GenerativeModel`) in 'process_df'.

To address this, you can tell Streamlit not to hash this argument by adding a
leading underscore to the argument's name in the function signature:

```
@st.cache_data
def process_df(_model, ...):
    ...
```

File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
File "C:\Users\p90022569\Downloads\score\a1c.py", line 468, in <module>
    out_df = process_df(df, col_name, model, use_llm)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 228, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 243, in _get_or_create_cached_value
    value_key = _make_value_key(
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 476, in _make_value_key
    raise UnhashableParamError(cache_type, func, arg_name, arg_value, exc)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 468, in _make_value_key
    update_hash(
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 169, in update_hash
    ch.update(hasher, val)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 345, in update
    b = self.to_bytes(obj)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 327, in to_bytes
    b = b"%s:%s" % (tname, self._to_bytes(obj))
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 650, in _to_bytes
    self.update(h, item)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 345, in update
    b = self.to_bytes(obj)
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 327, in to_bytes
    b = b"%s:%s" % (tname, self._to_bytes(obj))
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\caching\hashing.py", line 647, in _to_bytes
    raise UnhashableTypeError() from ex
