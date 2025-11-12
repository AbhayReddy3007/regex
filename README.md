AttributeError: 'dict' object has no attribute 'columns'

File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
File "C:\Users\p90022569\Downloads\score\app.py", line 239, in <module>
    if col_name_input and col_name_input in df.columns:
    
