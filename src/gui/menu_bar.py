"""
Menu bar GUI for MusicBar app using rumps.
"""
import rumps
import threading
import time
from typing import Optional
from audio.capture import AudioCapture, AudioCaptureError
from audio.monitor import AudioMonitor, AudioMetrics
from utils.config import config

class MusicBarApp(rumps.App):
    """Main menu bar application."""
    
    def __init__(self):
        # Initialize rumps app
        super(MusicBarApp, self).__init__(
            name=config.app_name,
            title="♪",  # White musical note for better visibility
            quit_button=None  # We'll add our own quit
        )
        
        # Initialize components
        self.audio_monitor = AudioMonitor(update_callback=self._on_audio_update)
        self.audio_capture = AudioCapture(callback=self.audio_monitor.process_audio)
        
        # Current capture mode
        self.use_system_audio = False
        self.current_capture = self.audio_capture
        
        # State management
        self.is_monitoring = False
        self._update_timer = None
        
        # Menu items
        self._setup_menu()
        
        if config.debug:
            print("MusicBarApp initialized")
        
        # Force menu bar to show icon immediately
        self._ensure_menu_bar_visible()
    
    def _ensure_menu_bar_visible(self):
        """Ensure the menu bar icon is visible."""
        try:
            # Force update the menu bar
            self.title = "♪"
            # Try to force a refresh
            import AppKit
            AppKit.NSApp.activateIgnoringOtherApps_(True)
            if config.debug:
                print("♪ Menu bar icon should now be visible")
        except Exception as e:
            if config.debug:
                print(f"Error ensuring menu bar visibility: {e}")
    
    def _setup_menu(self):
        """Set up the menu bar menu items."""
        # Status section
        self.status_item = rumps.MenuItem("🔇 Not monitoring")
        self.menu.add(self.status_item)
        self.menu.add(rumps.separator)
        
        # Audio metrics section - these will auto-update when monitoring
        self.volume_item = rumps.MenuItem("Volume: --")
        self.stereo_item = rumps.MenuItem("Stereo: --")  
        self.frequency_item = rumps.MenuItem("Peak Freq: --")
        
        self.menu.add(self.volume_item)
        self.menu.add(self.stereo_item)
        self.menu.add(self.frequency_item)
        self.menu.add(rumps.separator)
        
        # Controls section
        self.toggle_item = rumps.MenuItem(
            "Start Monitoring",
            callback=self._toggle_monitoring
        )
        self.menu.add(self.toggle_item)
        
        # Audio source selection - initialize first
        # (System audio capture removed, so no source selection needed)
        
        # Device selection (for microphone mode) - initialize first
        self.device_menu = rumps.MenuItem("Microphone Device")
        # Don't call _update_device_menu() here since it tries to clear()
        # Instead, add initial placeholder
        self.device_menu.add(rumps.MenuItem("Loading devices..."))
        self.menu.add(self.device_menu)
        
        self.menu.add(rumps.separator)
        
        # Settings and info
        self.settings_menu = rumps.MenuItem("Settings")
        self.settings_menu.add(rumps.MenuItem(
            f"Sample Rate: {config.audio.sample_rate}Hz",
            callback=None
        ))
        self.settings_menu.add(rumps.MenuItem(
            f"Channels: {config.audio.channels}",
            callback=None
        ))
        self.settings_menu.add(rumps.MenuItem(
            f"Buffer: {config.audio.chunk_size}",
            callback=None
        ))
        self.menu.add(self.settings_menu)
        
        # About and quit
        self.menu.add(rumps.MenuItem(
            f"About {config.app_name} v{config.version}",
            callback=self._show_about
        ))
        self.menu.add(rumps.MenuItem("Quit", callback=self._quit_app))
        
        # Now that menus are properly initialized, update the device menu
        self._safe_update_device_menu()
    
    def _select_source(self, source_type: str):
        """Select audio source type."""
        try:
            # Stop current monitoring
            was_monitoring = self.is_monitoring
            if self.is_monitoring:
                self._stop_monitoring()
            
            # Update source
            if source_type == "system":
                self.use_system_audio = True
                self.current_capture = self.audio_capture
                self.device_menu.title = "System Audio Device"
            else:
                self.use_system_audio = False
                self.current_capture = self.audio_capture
                self.device_menu.title = "Microphone Device"
            
            # Update menu checkmarks
            self._update_source_menu()
            
            # Restart monitoring if it was running
            if was_monitoring:
                self._start_monitoring()
            
            if config.debug:
                print(f"Switched to {source_type} audio source")
                
        except Exception as e:
            self._show_notification("Error", f"Failed to switch audio source: {e}")
    
    def _update_source_menu(self):
        """Update the audio source selection menu."""
        self.source_menu.clear()
        
        # Microphone option
        mic_title = "✓ Microphone Input" if not self.use_system_audio else "Microphone Input"
        self.source_menu.add(rumps.MenuItem(
            mic_title,
            callback=lambda sender: self._select_source("microphone")
        ))
        
        # System audio option
        sys_title = "✓ System Audio (Screen Recording)" if self.use_system_audio else "System Audio (Screen Recording)"
        self.source_menu.add(rumps.MenuItem(
            sys_title,
            callback=lambda sender: self._select_source("system")
        ))
    
    def _safe_update_device_menu(self):
        """Safely update the device menu without clearing uninitialized menus."""
        try:
            # Only clear if the menu is properly initialized
            if hasattr(self.device_menu, '_menu') and self.device_menu._menu is not None:
                self.device_menu.clear()
            else:
                # Menu not fully initialized, just remove existing items manually
                while len(self.device_menu._items) > 0:
                    item = self.device_menu._items[0]
                    self.device_menu.remove(item)
            
            # Microphone devices
            devices = AudioCapture.list_devices()
            current_device = self.audio_capture.get_device_info()
            
            if not devices:
                self.device_menu.add(rumps.MenuItem("No devices found"))
                return
            
            for device in devices:
                device_name = f"{device['name']} ({device['channels']}ch)"
                is_current = (
                    current_device.get('name') == device['name'] 
                    if 'name' in current_device else False
                )
                
                if is_current:
                    device_name = f"✓ {device_name}"
                
                item = rumps.MenuItem(
                    device_name,
                    callback=lambda sender, dev_id=device['id']: self._select_device(dev_id)
                )
                self.device_menu.add(item)
                
        except Exception as e:
            if config.debug:
                print(f"Error updating device menu: {e}")
            # Add fallback message
            self.device_menu.add(rumps.MenuItem("Device menu error"))
    
    def _update_device_menu(self):
        """Update the audio device selection menu."""
        # Use the safe version
        self._safe_update_device_menu()
    
    def _select_device(self, device_id: int):
        """Select a new audio device."""
        try:
            # Stop current monitoring
            was_monitoring = self.is_monitoring
            if self.is_monitoring:
                self._stop_monitoring()
            
            # Update device configuration
            config.audio.device = device_id
            self.audio_capture = AudioCapture(callback=self.audio_monitor.process_audio)
            self.current_capture = self.audio_capture
            
            # Restart monitoring if it was running
            if was_monitoring:
                self._start_monitoring()
            
            # Update menu
            self._update_device_menu()
            
            if config.debug:
                print(f"Switched to audio device: {device_id}")
                
        except Exception as e:
            self._show_notification("Error", f"Failed to switch device: {e}")
    
    def _toggle_monitoring(self, sender):
        """Toggle audio monitoring on/off."""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start audio monitoring."""
        try:
            # Start microphone (or loopback) capture
            self.audio_capture.start()
            
            self.is_monitoring = True
            
            # Update UI
            source_name = "Microphone"
            self.title = "♪📊"
            self.toggle_item.title = "Stop Monitoring"
            self.status_item.title = f"♪ Monitoring {source_name}..."
            
            # Start update timer for live updates
            self._start_update_timer()
            
            if config.debug:
                print(f"Audio monitoring started ({source_name})")
                
        except (AudioCaptureError) as e:
            self._show_notification("Audio Error", str(e))
            if config.debug:
                print(f"Failed to start monitoring: {e}")
    
    def _stop_monitoring(self):
        """Stop audio monitoring."""
        # Stop both capture methods
        self.audio_capture.stop()
        
        self.is_monitoring = False
        
        # Update UI
        self.title = "♪"
        self.toggle_item.title = "Start Monitoring"
        self.status_item.title = "🔇 Not monitoring"
        
        # Reset display
        self.volume_item.title = "Volume: --"
        self.stereo_item.title = "Stereo: --"
        self.frequency_item.title = "Peak Freq: --"
        
        # Stop update timer
        self._stop_update_timer()
        
        if config.debug:
            print("Audio monitoring stopped")
    
    def _start_update_timer(self):
        """Start the UI update timer."""
        self._stop_update_timer()  # Ensure no duplicate timers
        
        if config.debug:
            print(f"Starting update timer with interval: {config.ui.update_interval}")
        
        self._update_timer = rumps.Timer(
            self._refresh_display,
            config.ui.update_interval
        )
        self._update_timer.start()
        
        if config.debug:
            print("✅ Update timer started")
    
    def _stop_update_timer(self):
        """Stop the UI update timer."""
        if self._update_timer:
            if config.debug:
                print("Stopping update timer")
            self._update_timer.stop()
            self._update_timer = None
    
    def _refresh_display(self, sender=None):
        """Refresh the display with current audio metrics - can be called by timer or manually."""
        if not self.is_monitoring:
            return
        
        try:
            # Get current metrics
            metrics = self.audio_monitor.get_metrics()
            volume_db = self.audio_monitor.get_volume_db()
            stereo_info = self.audio_monitor.get_stereo_info()
            
            # Update volume display
            if metrics.overall_volume > 0.001:  # Avoid showing tiny values
                volume_text = f"Volume: {volume_db['overall_db']:.1f}dB"
                if config.debug and sender is None:  # Only debug on timer calls, not manual
                    print(f"Volume: {volume_db['overall_db']:.1f}dB")
                
                # Update title icon based on volume level
                if volume_db['overall_db'] > -10:
                    new_title = "♪🔊"
                elif volume_db['overall_db'] > -20:
                    new_title = "♪🔉"
                elif volume_db['overall_db'] > -40:
                    new_title = "♪🔈"
                else:
                    new_title = "♪📊"
            else:
                volume_text = "Volume: Silence"
                new_title = "♪📊"
            
            # Update stereo display
            stereo_text = (
                f"Stereo: {stereo_info['balance_description']} "
                f"({stereo_info['width_description']})"
            )
            
            # Update frequency display
            if metrics.peak_frequency > 0:
                if metrics.peak_frequency < 1000:
                    freq_str = f"{metrics.peak_frequency:.0f}Hz"
                else:
                    freq_str = f"{metrics.peak_frequency/1000:.1f}kHz"
                freq_text = f"Peak Freq: {freq_str}"
            else:
                freq_text = "Peak Freq: --"
            
            # Apply all updates at once
            self.volume_item.title = volume_text
            self.stereo_item.title = stereo_text
            self.frequency_item.title = freq_text
            self.title = new_title
            
        except Exception as e:
            if config.debug:
                print(f"Error refreshing display: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_audio_update(self, metrics: AudioMetrics):
        """Called when new audio metrics are available."""
        # This is called from the audio thread, so we don't update UI directly
        # The UI update timer handles the display updates
        pass
    
    def _show_about(self, sender):
        """Show about dialog."""
        message = (
            f"{config.app_name} v{config.version}\n\n"
            "Real-time audio monitoring and visualization\n"
            "for macOS menu bar.\n\n"
            "Features:\n"
            "• Microphone input monitoring\n"
            "• Real-time volume and stereo analysis\n"
            "• Frequency domain analysis\n\n"
            "© 2024 MusicBar Project"
        )
        rumps.alert(
            title=f"About {config.app_name}",
            message=message,
            ok="OK"
        )
    
    def _show_notification(self, title: str, message: str):
        """Show a system notification."""
        if config.ui.show_notifications:
            rumps.notification(
                title=title,
                subtitle=config.app_name,
                message=message
            )
    
    def _quit_app(self, sender):
        """Quit the application."""
        if self.is_monitoring:
            self._stop_monitoring()
        rumps.quit_application()

def run_app():
    """Entry point to run the menu bar app."""
    try:
        app = MusicBarApp()
        if config.debug:
            print("♪ MusicBar starting... Look for the ♪ icon in your menu bar!")
        app.run()
    except KeyboardInterrupt:
        if config.debug:
            print("Application interrupted by user")
    except Exception as e:
        if config.debug:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        else:
            rumps.alert("Error", f"Application error: {e}")

if __name__ == "__main__":
    run_app() 