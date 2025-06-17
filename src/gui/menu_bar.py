"""
Menu bar GUI for MusicBar app using pure AppKit.
This approach replaces the `rumps` library for greater control and flexibility.
"""

import AppKit
import objc
import numpy as np
from PyObjCTools import AppHelper

from audio.capture import AudioCapture, AudioCaptureError
from audio.monitor import AudioMonitor, AudioMetrics
from utils.config import config

class UITimer(AppKit.NSObject):
    """A dedicated controller for the NSTimer to avoid conflicts with object lifecycles."""
    _controller: 'MusicBarController' = objc.ivar()
    _timer: AppKit.NSTimer = objc.ivar()

    @objc.selector
    def initWithController_(self, controller: 'MusicBarController'):
        self = objc.super(UITimer, self).init()
        if self is None: return None
        self._controller = controller
        self._timer = None
        return self

    @objc.python_method
    def start(self, interval: float):
        self.stop()
        self._timer = AppKit.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, b'tick:', None, True
        )
        AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._timer, AppKit.NSRunLoopCommonModes
        )
        if config.debug:
            print(f"UITimer started for common modes with interval: {interval}")

    @objc.python_method
    def stop(self):
        if self._timer and self._timer.isValid():
            self._timer.invalidate()
            self._timer = None
            if config.debug:
                print("UITimer stopped")

    @objc.selector
    def tick_(self, timer: AppKit.NSTimer):
        if self._controller:
            self._controller.refresh_display()


class StereoImageView(AppKit.NSView):
    """A custom view to visualize the stereo image."""
    _balance = objc.ivar.double()
    _width = objc.ivar.double()

    def initWithFrame_(self, frame):
        self = objc.super(StereoImageView, self).initWithFrame_(frame)
        if self:
            self._balance = 0.0
            self._width = 0.0
        return self

    def drawRect_(self, dirtyRect):
        AppKit.NSColor.clearColor().set()
        AppKit.NSRectFill(self.bounds())

        bounds_inset = self.bounds().insetBy_dx_dy_(10, 10)
        
        path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(bounds_inset)
        AppKit.NSColor.grayColor().set()
        path.setLineWidth_(1)
        path.stroke()

        center_x = self.bounds().size.width / 2
        center_y = self.bounds().size.height / 2
        
        radius = (bounds_inset.size.width / 2) * self._width
        angle = (1.0 - self._balance) * (np.pi / 2)

        dot_x = center_x + radius * np.cos(angle) - 5
        dot_y = center_y + radius * np.sin(angle) - 5
        
        dot_rect = AppKit.NSMakeRect(dot_x, dot_y, 10, 10)
        dot_path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
        
        AppKit.NSColor.whiteColor().set()
        dot_path.fill()

    @objc.python_method
    def update_stereo_image(self, balance: float, width: float):
        if self._balance != balance or self._width != width:
            self._balance = balance
            self._width = width
            self.setNeedsDisplay_(True)

class MusicBarController(AppKit.NSObject):
    """Main controller for the application. Manages UI and audio components."""
    
    def init(self):
        self = objc.super(MusicBarController, self).init()
        if not self: return None

        self.audio_monitor = AudioMonitor(update_callback=self._on_audio_update)
        self.audio_capture = AudioCapture(callback=self.audio_monitor.process_audio)
        self.is_monitoring = False
        
        self.setup_status_item()
        self.setup_menu()
        self.setup_popover()
        
        self._ui_timer = UITimer.alloc().initWithController_(self)
        self.status_item.setMenu_(self.menu)

        if config.debug:
            print("MusicBarController initialized")
        return self

    def setup_status_item(self):
        self.status_item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        button = self.status_item.button()
        button.setTitle_("♪")
        button.setAction_(b'togglePopover:')
        button.setTarget_(self)

    def setup_popover(self):
        self.popover = AppKit.NSPopover.alloc().init()
        self.popover.setBehavior_(AppKit.NSPopoverBehaviorTransient)
        self.popover.setAnimates_(True)
        
        self.stereo_view = StereoImageView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 150, 150))
        vc = AppKit.NSViewController.alloc().init()
        vc.setView_(self.stereo_view)
        self.popover.setContentViewController_(vc)

    def setup_menu(self):
        self.menu = AppKit.NSMenu.alloc().init()

        noop_action = b'noop:'
        self.status_item_menu = self.create_menu_item("🔇 Not monitoring", noop_action, target=self)
        self.volume_item = self.create_menu_item("Volume: --", noop_action, target=self)
        self.stereo_item = self.create_menu_item("Stereo: --", noop_action, target=self)
        self.bpm_item = self.create_menu_item("BPM: --", noop_action, target=self)
        self.chord_item = self.create_menu_item("Chord: --", noop_action, target=self)
        self.toggle_item = self.create_menu_item("Start Monitoring", b'toggleMonitoring:', target=self)
        
        self.device_menu = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Microphone Device", None, "")
        self.device_submenu = AppKit.NSMenu.alloc().init()
        self.device_menu.setSubmenu_(self.device_submenu)

        self.menu.addItem_(self.status_item_menu)
        self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self.menu.addItem_(self.volume_item)
        self.menu.addItem_(self.stereo_item)
        self.menu.addItem_(self.bpm_item)
        self.menu.addItem_(self.chord_item)
        self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self.menu.addItem_(self.toggle_item)
        self.menu.addItem_(self.device_menu)
        self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self.menu.addItem_(self.create_menu_item(f"About {config.app_name}", b'showAbout:', target=self))
        self.menu.addItem_(self.create_menu_item("Quit", b'quitApp:', 'q', target=self))

        self.update_device_menu()

    def create_menu_item(self, title, action=None, key='', target=None):
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        if target:
            item.setTarget_(target)
        return item

    def update_device_menu(self):
        self.device_submenu.removeAllItems()
        devices = AudioCapture.list_devices()
        current_device_id = self.audio_capture.device

        for device in devices:
            title = f"{device['name']} ({device['channels']}ch)"
            if device['id'] == current_device_id:
                title = f"✓ {title}"
            item = self.create_menu_item(title, b'selectDevice:', target=self)
            item.setRepresentedObject_(device['id'])
            self.device_submenu.addItem_(item)

    @objc.selector
    def selectDevice_(self, sender):
        device_id = sender.representedObject()
        if device_id == self.audio_capture.device: return

        was_monitoring = self.is_monitoring
        if was_monitoring: self.stop_monitoring()
        
        config.audio.device = device_id
        self.audio_capture = AudioCapture(callback=self.audio_monitor.process_audio)
        
        if was_monitoring: self.start_monitoring()
        self.update_device_menu()

    @objc.selector
    def togglePopover_(self, sender):
        if self.popover.isShown():
            self.popover.close()
        else:
            self.popover.showRelativeToRect_ofView_preferredEdge_(sender.bounds(), sender, AppKit.NSMinYEdge)

    @objc.selector
    def toggleMonitoring_(self, sender):
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    @objc.python_method
    def start_monitoring(self):
        try:
            self.audio_capture.start()
            self.is_monitoring = True
            self.toggle_item.setTitle_("Stop Monitoring")
            self.status_item_menu.setTitle_(f"♪ Monitoring Microphone...")
            self._ui_timer.start(config.ui.update_interval)
            if config.debug: print("Audio monitoring started")
        except AudioCaptureError as e:
            self.show_alert("Audio Error", str(e))
            if config.debug: print(f"Failed to start monitoring: {e}")

    @objc.python_method
    def stop_monitoring(self):
        if not self.is_monitoring: return
        self.audio_capture.stop()
        self.is_monitoring = False
        self._ui_timer.stop()
        self.toggle_item.setTitle_("Start Monitoring")
        self.status_item_menu.setTitle_("🔇 Not monitoring")
        self.status_item.button().setTitle_("♪")
        self.volume_item.setTitle_("Volume: --")
        self.stereo_item.setTitle_("Stereo: --")
        self.bpm_item.setTitle_("BPM: --")
        self.chord_item.setTitle_("Chord: --")
        if self.popover.isShown(): self.popover.close()
        if config.debug: print("Audio monitoring stopped")

    @objc.python_method
    def refresh_display(self):
        if not self.is_monitoring: return
        try:
            metrics = self.audio_monitor.get_metrics()
            avg_volume = np.mean(list(metrics.volume_history)) if metrics.volume_history else 0.0

            if avg_volume > 0.001:
                volume_db_val = 20 * np.log10(avg_volume)
                self.volume_item.setTitle_(f"Volume: {volume_db_val:.1f}dB")
                if volume_db_val > -10: title = "♪🔊"
                elif volume_db_val > -20: title = "♪🔉"
                elif volume_db_val > -40: title = "♪🔈"
                else: title = "♪📊"
            else:
                self.volume_item.setTitle_("Volume: Silence")
                title = "♪📊"
            self.status_item.button().setTitle_(title)

            stereo_info = self.audio_monitor.get_stereo_info()
            self.stereo_item.setTitle_(f"Stereo: {stereo_info['balance_description']} ({stereo_info['width_description']})")
            
            if self.stereo_view:
                self.stereo_view.update_stereo_image(stereo_info.get('balance', 0.0), stereo_info.get('width', 0.0))

            self.bpm_item.setTitle_(f"BPM: {metrics.bpm:.1f}" if metrics.bpm > 0 else "BPM: --")
            self.chord_item.setTitle_(f"Chord: {metrics.chord}")

        except Exception as e:
            if config.debug: print(f"Error refreshing display: {e}")

    @objc.python_method
    def _on_audio_update(self, metrics: AudioMetrics):
        pass # The UI timer handles all display updates

    @objc.selector
    def showAbout_(self, sender):
        self.show_alert(f"About {config.app_name}", f"{config.app_name} v{config.version}")

    @objc.python_method
    def show_alert(self, title, message):
        alert = AppKit.NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.setAlertStyle_(AppKit.NSAlertStyleInformational)
        alert.runModal()
        
    @objc.selector
    def quitApp_(self, sender):
        self.stop_monitoring()
        AppKit.NSApplication.sharedApplication().terminate_(None)

    @objc.selector
    def noop_(self, sender):
        """A no-op action to ensure menu items are enabled and can be redrawn."""
        pass

class AppDelegate(AppKit.NSObject):
    _controller: MusicBarController = objc.ivar()

    def applicationDidFinishLaunching_(self, notification):
        self._controller = MusicBarController.alloc().init()
        if config.debug:
            print("AppDelegate finished launching, controller created.")

    def applicationWillTerminate_(self, notification):
        if self._controller:
            self._controller.stop_monitoring()
        if config.debug:
            print("Application will terminate.")

def run_app():
    """Entry point to run the menu bar app."""
    try:
        if config.debug: print("Starting AppKit application...")
        
        app = AppKit.NSApplication.sharedApplication()
        delegate = AppDelegate.alloc().init()
        app.setDelegate_(delegate)
        
        AppHelper.runEventLoop()

    except KeyboardInterrupt:
        if config.debug: print("Application interrupted by user.")
    except Exception as e:
        if config.debug: print(f"Application error: {e}")
        # Consider a native alert for the user here.

if __name__ == "__main__":
    run_app() 