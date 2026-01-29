; restart_routine.ahk
#NoEnv
SendMode Input
SetWorkingDir %A_ScriptDir%

; ==========================================================
; FORCE RUN AS ADMINISTRATOR
; ==========================================================
if not A_IsAdmin
{
    Run *RunAs "%A_ScriptFullPath%"  ; Restart this script as Admin
    ExitApp
}

; ==========================================================
; THE SEQUENCE
; ==========================================================

; 1. Force Focus on Game
IfWinExist, NBA 2K25
{
    WinActivate
    Sleep, 500
}

; 2. Sequence: Esc -> Left -> Down -> 2 -> Down -> 2

; Open Menu
Send {Esc down}
Sleep 100
Send {Esc up}
Sleep 2000 ; Wait for menu animation

; Move Left
Send {Left down}
Sleep 100
Send {Left up}
Sleep 500

; Move Down (to highlight Rematch?)
Send {Down down}
Sleep 100
Send {Down up}
Sleep 500

; SELECT REMATCH (Press 2)
Send {2 down}
Sleep 100
Send {2 up}
Sleep 1500 ; Wait for "Are you sure?" dialog

; Move Down (to highlight Yes)
Send {Down down}
Sleep 100
Send {Down up}
Sleep 500

; SELECT YES (Press 2)
Send {2 down}
Sleep 100
Send {2 up}

ExitApp