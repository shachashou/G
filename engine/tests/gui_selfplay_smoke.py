"""Desktop teacher-free mode stays usable when the teacher is unavailable."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("gui", "tools", "trainer", "engine/tests")]
from PySide6.QtWidgets import QApplication
import gomoku_gui
import rapfi_teacher
from training_task import TeacherDataSource
from training_task_panel import TrainingTaskPanel
from selfplay_task_tests import make_incumbent
from value_fixture import build_pool


def wait(app, predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(.01)
    return False


def main():
    exe = str(Path(sys.argv[1]).resolve())
    app = QApplication.instance() or QApplication([sys.argv[0]])
    with tempfile.TemporaryDirectory(prefix="gui-selfplay-") as raw:
        root = Path(raw)
        panel = TrainingTaskPanel(exe, root/'empty.sqlite3', root/'missing-registry',
            root/'empty-evaluations', root/'empty-tasks', lambda: True)
        assert panel.spin_batch.value() == 2048
        assert panel.spin_pending.value() == 8192
        assert panel.make_config().training_sample_limit == 32768
        panel.btn_quick.click()
        assert panel.make_config().collector_workers == 1
        busy = []
        panel.busyChanged.connect(busy.append)
        with patch('training_task.TrainingTask', side_effect=AssertionError('unready task created')):
            for mode in ('transition', 'selfplay'):
                panel.combo_mode.setCurrentIndex(panel.combo_mode.findData(mode))
                panel.btn_start.click()
                assert panel.thread is None and panel.task is None
                assert '先建立正式版本 v0' in panel.status.text(), panel.status.text()
                assert not panel.btn_bootstrap.isHidden()
                assert not (root/'empty-tasks').exists()
                panel.btn_bootstrap.click()
                assert panel.combo_mode.currentData() == 'teacher'
                assert panel.task is None  # Explicit mode selection does not start a teacher.
            broken = root/'broken-registry'
            broken.mkdir()
            (broken/'state.json').write_text('{broken', encoding='utf-8')
            panel.registry = broken
            panel.combo_mode.setCurrentIndex(panel.combo_mode.findData('selfplay'))
            panel.btn_start.click()
            assert '正式版本校验失败' in panel.status.text()
            assert panel.btn_bootstrap.isHidden()
            assert not busy
        panel.shutdown()
        panel.close()
        print('missing/invalid incumbent -> actionable pre-start guidance, no failed task: passed')
        registry = make_incumbent(root, exe)
        pool = build_pool(root / "mixed.sqlite3", validation_games=1)
        missing = rapfi_teacher.InstallCheck(False, root/'missing-teacher', None, ['teacher disabled for acceptance'])
        with patch.object(rapfi_teacher, 'check_install', return_value=missing), \
                patch.object(TeacherDataSource, '__init__', side_effect=AssertionError('teacher initialization forbidden')):
            window = gomoku_gui.MainWindow(exe,20,pool=pool,registry=registry,evaluations=root/'evaluations')
            window.show()
            panel = window.training_task_panel
            panel.btn_quick.click()
            assert '不可用' in window.teacher_label.text()
            panel.combo_mode.setCurrentIndex(panel.combo_mode.findData('transition'))
            assert '还不是完全无教师' in panel.hint.text()
            assert panel.spin_teacher_weight.isEnabled()
            # Recheck at click time, even if the model existed when mode was selected.
            panel.registry = root/'missing-registry'
            panel.btn_start.click()
            assert panel.thread is None and window._phase != 'continuous'
            assert '先建立正式版本 v0' in panel.status.text()
            panel.registry = registry
            panel.combo_mode.setCurrentIndex(panel.combo_mode.findData('selfplay'))
            assert '教师标签被排除' in panel.hint.text()
            assert not panel.spin_teacher_weight.isEnabled()
            panel.edit_output.setText(str(root/'tasks'))
            panel.spin_min_train.setValue(1)
            panel.spin_new.setValue(1)
            panel.spin_steps.setValue(4)
            panel.spin_batch.setValue(4)
            panel.spin_rounds.setValue(1)
            panel.spin_pairs.setValue(1)
            panel.spin_ms.setValue(10)
            panel.spin_lr.setValue(.00001)
            panel.combo_device.setCurrentText('cpu')
            config = replace(panel.make_config(),validation_games=1,max_pending_samples=4,
                min_round_interval_s=0,max_games=15,seed=32,
                selfplay={'opening_plies':2,'epsilon':.05,'movetime_ms':10,'vcf_policy':'off','filter_vcf':False},
                evaluation={'pairs':1,'movetime_ms':10,'vcf_policy':'off'})
            panel.make_config = lambda: config
            panel.btn_start.click()
            assert window._phase == 'continuous'
            assert wait(app,lambda: panel.thread is None),panel.status.text()
            result = panel.last_result
            assert result and result['status']=='completed',result
            assert result['rounds'] and result['rounds'][0]['evaluation'],result
            assert result['rounds'][0]['data_identity']['label_sources']==['selfplay_result']
            assert not result['rounds'][0]['evaluation']['publication']
            assert 'selfplay' in panel.metrics.toPlainText()
            assert '当前正式版本 v0' in window.eval_label.text()
            print('teacher missing -> mixed pool -> selfplay only -> candidate/history evaluation: passed')
            panel.make_config = lambda: replace(config,max_rounds=10)
            panel.btn_start.click()
            panel.btn_stop.click()
            assert wait(app,lambda: panel.thread is None)
            assert panel.last_result['status']=='stopped'
            assert panel.combo_mode.currentData()=='selfplay'
            panel.combo_mode.setCurrentIndex(panel.combo_mode.findData('teacher'))
            assert panel.combo_mode.currentData()=='teacher'
            assert 'Rapfi' in panel.hint.text()
            print('stop retains incumbent and explicit mode switch: passed')
            window.close()
            app.processEvents()
    print('gui_selfplay_smoke: all passed')
    return 0


if __name__=='__main__':
    sys.exit(main())
