"""CPU-only checks for GPU campaign ownership and checkpoint identity controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.replicate_queue import Campaign,cached_model


class QueueControls(unittest.TestCase):
    def test_cleanup_terminates_only_owned_process_group(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign=Campaign.__new__(Campaign)
            campaign.out=Path(folder);campaign.status={'active_processes':[]}
            campaign.children=[];campaign.known_roots=set();campaign.logs=[];campaign.locks=[]
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            campaign.children.append(owned)
            try:
                campaign.cleanup()
                self.assertIsNotNone(owned.poll())
                self.assertIsNone(unrelated.poll())
                self.assertEqual(campaign.status['active_processes'],[])
            finally:
                if owned.poll() is None:owned.terminate();owned.wait()
                unrelated.terminate();unrelated.wait()

    def test_missing_pinned_revision_never_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder)
            snapshot=cache/'models--example--model/snapshots/available'
            snapshot.mkdir(parents=True)
            (snapshot/'config.json').write_text('{}')
            (snapshot/'model.safetensors').write_bytes(b'test fixture')
            with patch('scripts.replicate_queue.CACHE',cache):
                self.assertEqual(cached_model('example/model')[1],'available')
                with self.assertRaises(FileNotFoundError):cached_model('example/model','missing-pinned-revision')

    def test_immediate_stop_marker_interrupts(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign=Campaign.__new__(Campaign);campaign.out=Path(folder)
            (campaign.out/'STOP_NOW').touch()
            with self.assertRaises(KeyboardInterrupt):campaign.stop_requested()


if __name__=='__main__':unittest.main()
