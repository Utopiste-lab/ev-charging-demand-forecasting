"""Unit checks for boundaries, masks, timestamps and purging."""
import csv
import gzip
import io
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import yaml
from src.preprocessing.parser import parse_bytes
from src.preprocessing.resampling import grid,prefix,future_target,SECOND
from src.preprocessing.dataset_split import split,DAY

CFG=yaml.safe_load((ROOT/"config/preprocessing.yaml").read_text())


def payload(rows):
    return gzip.compress((",Charging Current (A),Voltage (V)\n"+"\n".join(rows)+"\n").encode(),mtime=0)


class TestPreprocessing(unittest.TestCase):
    def test_sorted_elapsed_malformed_duplicate_and_missing(self):
        rows=["2020-01-01T00:00:06+05:30,NaN,208","2020-01-01T00:00:00+05:30,0,208",
              "2020-01-01T00:00:05+05:30,7.5,210","2020-01-01T00:00:05+05:30,8,210",
              "2020-01-01T00:00:07+05:30,inf,208","bad,3,208","malformed"]
        out=io.StringIO()
        p=parse_bytes(payload(rows),200000,csv.writer(out))
        self.assertEqual(p.info["first_telemetry"],"2020-01-01T00:00:00+05:30")
        self.assertEqual(p.info["offset_set"],"+0530")
        self.assertEqual(p.info["original_rows"],7)
        self.assertEqual(p.info["parsed_rows"],5)
        self.assertEqual(p.info["duplicate_timestamp_rows"],1)
        self.assertEqual(p.info["conflicting_current_times"],1)
        self.assertEqual(p.info["missing_current_rows"],1)
        self.assertEqual(p.info["invalid_current_rows"],1)
        self.assertEqual(p.info["invalid_timestamp_rows"],1)
        self.assertEqual(p.info["malformed_rows"],1)
        self.assertIn("malformed_row",out.getvalue())
        np.testing.assert_array_equal(p.values,[0.])
        np.testing.assert_array_equal(p.elapsed_us,[0])

    def test_gap_rules_and_zeros(self):
        t=np.array([0,10,30,60,120,210],dtype=np.int64)*SECOND
        y=np.array([0.,6.,2.,4.,5.,8.])
        g=grid(t,y,0,24,5,CFG)
        self.assertTrue(g.observed[0])
        self.assertEqual(g.values[0],0)
        self.assertTrue(g.interpolated[1])
        self.assertEqual(g.values[1],3)
        self.assertFalse(g.interpolated[3:6].any())
        self.assertEqual(g.values[3],0)
        self.assertFalse(g.observed[3])
        # A single empty 30s bin whose raw endpoints are 60s apart stays missing.
        w=grid(t,y,30*SECOND,5,30,CFG)
        self.assertFalse(w.interpolated.any())

    def test_prefix_and_future_boundary_perturbation(self):
        base=["2020-01-01T00:00:00+00:00,0,208","2020-01-01T00:04:59+00:00,11,208"]
        tail=["2020-01-01T00:05:00+00:00,200,208","2020-01-01T00:05:05+00:00,13,208",
              "2020-01-01T00:10:00+00:00,999,208"]
        a=parse_bytes(payload(base+tail),200000)
        b=parse_bytes(payload(base+[row.replace(",13,",",88,") for row in tail]),200000)
        for step in (5,10,30):
            for minutes in (5,10):
                x,qa=prefix(a,minutes,step,CFG)
                self.assertEqual(len(x.values),60*minutes//step)
                if minutes==5:
                    z,_=prefix(b,minutes,step,CFG)
                    for key in ("values","observed","interpolated"):
                        np.testing.assert_array_equal(getattr(x,key),getattr(z,key))
                    self.assertLess(x.values.max(),200)
                    y,q=future_target(a,minutes,10,step,CFG)
                    self.assertEqual(y.values[0],13)
                    self.assertFalse(y.interpolated.any())
                    self.assertFalse(q["span_reaches_window"])
                    self.assertFalse(q["strict"])
                    self.assertLess(y.observed.sum(),len(y.values))
                else:
                    self.assertLess(x.values.max(),999)

    def test_cross_boundary_group_and_long_footprint_purged(self):
        data=[]
        for i in range(20):
            t=(100+i)*DAY+10
            data.append(dict(record_id=f"r{i}",site="Caltech" if i%2 else "JPL",cohort="representative",
                             first_timestamp_us=t,last_timestamp_us=t+500,clock_admissible=True,duplicate_group_id=f"g{i}"))
        data[0]["duplicate_group_id"]=data[18]["duplicate_group_id"]
        data[1]["last_timestamp_us"]=999*DAY
        out,_=split(pd.DataFrame(data),CFG)
        s=out.set_index("record_id")
        self.assertEqual(s.at["r0","split_exclusion_reason"],"relationship_group_crosses_cutoff")
        self.assertEqual(s.at["r18","partition"],"excluded")
        self.assertEqual(s.at["r1","split_exclusion_reason"],"observed_record_crosses_cutoff")
        self.assertTrue(out.query('partition in ["train","validation","test"]').groupby("duplicate_group_id").partition.nunique().le(1).all())


if __name__=="__main__":
    unittest.main(verbosity=2)

