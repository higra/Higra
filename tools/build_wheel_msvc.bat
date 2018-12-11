REM Simple script to build pypi wheels on windows
REM Very dirty: don't use it
REM Really i'm serious don't use it

@echo off
set pyversions=3.4 3.5 3.6 3.7


set wheellocation=.\wheelhouse
set tmpfolder=.\tmp-build-wheel

if exist %wheellocation% RMDIR "%wheellocation%" /s /q
MKDIR "%wheellocation%"
CALL :NORMALIZEPATH "%wheellocation%"
SET wheellocation=%RETVAL%

if exist "%tmpfolder%" RMDIR "%tmpfolder%" /s /q
MKDIR "%tmpfolder%"
CALL :NORMALIZEPATH "%tmpfolder%"
SET tmpfolder=%RETVAL%

cd %tmpfolder%

FOR %%p IN (%pyversions%) DO (
	CALL :PROCESS_VERSION %%p
)

cd ..
RMDIR "%tmpfolder%" /s /q
echo Finished, wheels are in %wheellocation%
echo To upload to pypi: twine upload %wheellocation%\*.whl

:: ========== FUNCTIONS ==========
EXIT /B

:PROCESS_VERSION
	set pyversion=%~1
	set env_name=build-wheel-py%~1
	echo Preparing environment %env_name%
	call conda remove --name %env_name% --all --yes || exit /b
	call conda create -n %env_name% python=%pyversion% pip numpy cmake boost --yes || exit /b
	call conda activate %env_name%  || exit /b
	call python -m pip install --upgrade pip || exit /b
	
	echo Building project
	call git clone https://github.com/PerretB/Higra.git || exit /b
	cd Higra
	call pip wheel . || exit /b
	
	echo Testing project
	for  %%s in (higra*whl) do (
		call pip install %%s || exit /b
	)
	cd ..
	call python -c "import unittest;result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('Higra\\test\\python\\'));exit(0 if result.wasSuccessful() else 1)" || exit /b

	echo Copying wheel
	xcopy .\Higra\higra*whl %wheellocation%
	
	echo Cleaning environment
	RMDIR Higra /s /q
	call conda deactivate || exit /b
	call conda remove --name %env_name% --all --yes || exit /b
	
	echo done
	EXIT /B

:NORMALIZEPATH
  SET RETVAL=%~dpfn1
  EXIT /B
