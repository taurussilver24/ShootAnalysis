; NBA 2K25 Auto-Rematch Script
; Press F9 to start/stop automation
; Press F10 to execute one sequence manually
; Press ESC to exit script

#NoEnv
#SingleInstance Force
SetWorkingDir %A_ScriptDir%

; ============================================================================
; CONFIGURATION - ADJUST THESE TO MATCH YOUR GAME
; ============================================================================

; Process name of the game
GAME_PROCESS := "NBA2K25.exe"

; Key navigation sequence
KEY_OPEN_MENU := "{Esc}"           ; Opens the quit menu
KEY_TO_QUIT := "{Left}"            ; Navigate to QUIT (one tap left)
KEY_TO_REMATCH := "{Down}"         ; From QUIT to REMATCH (one tap down)
KEY_TO_YES := "{Down}"             ; From REMATCH to YES (one tap down)  
KEY_SELECT := "{Enter}"            ; Confirm selection (Enter or 2)

; Timing (in milliseconds)
KEY_DELAY := 500                   ; Time between key presses
MENU_TRANSITION := 1500            ; Time to wait for menus to appear
COOLDOWN_TIME := 30000             ; Time between full sequences (30 seconds)

; ============================================================================
; GLOBAL VARIABLES
; ============================================================================

isRunning := false
totalSequences := 0

; ============================================================================
; HOTKEYS
; ============================================================================

F9::
    isRunning := !isRunning
    if (isRunning) {
        TrayTip, NBA 2K25 Auto-Rematch, Automation STARTED, 2, 1
        SetTimer, AutomationLoop, 1000
    } else {
        TrayTip, NBA 2K25 Auto-Rematch, Automation STOPPED, 2, 1
        SetTimer, AutomationLoop, Off
    }
return

F10::
    TrayTip, NBA 2K25 Auto-Rematch, Executing ONE sequence manually, 2, 1
    ExecuteSequence()
return

Esc::
    TrayTip, NBA 2K25 Auto-Rematch, Script terminated, 2, 1
    Sleep, 500
    ExitApp
return

; ============================================================================
; MAIN AUTOMATION LOOP
; ============================================================================

AutomationLoop:
    if (!isRunning)
        return
    
    ; Check if game is running
    if (!IsGameRunning()) {
        TrayTip, NBA 2K25 Auto-Rematch, Waiting for game to launch..., 2, 1
        return
    }
    
    ; Execute the rematch sequence
    ExecuteSequence()
    
    ; Wait for cooldown
    TrayTip, NBA 2K25 Auto-Rematch, Cooldown %COOLDOWN_TIME%ms before next sequence, 2, 1
    Sleep, %COOLDOWN_TIME%
return

; ============================================================================
; CORE FUNCTIONS
; ============================================================================

IsGameRunning() {
    Process, Exist, %GAME_PROCESS%
    return ErrorLevel
}

FocusGame() {
    ; Find and activate game window
    WinActivate, ahk_exe %GAME_PROCESS%
    Sleep, 300
}

ExecuteSequence() {
    global totalSequences
    totalSequences++
    
    TrayTip, NBA 2K25 Auto-Rematch, Executing sequence #%totalSequences%, 2, 1
    
    ; Make sure game has focus
    FocusGame()
    Sleep, 500
    
    ; Step 1: Open menu
    Send, %KEY_OPEN_MENU%
    Sleep, %MENU_TRANSITION%
    
    ; Step 2: Navigate to QUIT
    Send, %KEY_TO_QUIT%
    Sleep, %KEY_DELAY%
    
    ; Step 3: Select QUIT
    Send, %KEY_SELECT%
    Sleep, %MENU_TRANSITION%
    
    ; Step 4: Navigate to REMATCH
    Send, %KEY_TO_REMATCH%
    Sleep, %KEY_DELAY%
    
    ; Step 5: Select REMATCH
    Send, %KEY_SELECT%
    Sleep, %MENU_TRANSITION%
    
    ; Step 6: Navigate to YES
    Send, %KEY_TO_YES%
    Sleep, %KEY_DELAY%
    
    ; Step 7: Confirm YES
    Send, %KEY_SELECT%
    
    TrayTip, NBA 2K25 Auto-Rematch, Sequence #%totalSequences% complete!, 2, 1
}

; ============================================================================
; STARTUP
; ============================================================================

TrayTip, NBA 2K25 Auto-Rematch, Script loaded! Press F9 to start`, F10 for manual`, ESC to exit, 5, 1
return