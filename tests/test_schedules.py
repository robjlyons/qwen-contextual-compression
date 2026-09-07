from pathlib import Path
import numpy as np
from end_to_end.schedules import active_parameter_accounting,expand_schedule,load_schedules

def test_linear_and_nearest_interpolation_are_exactly_64_layers():
 spec={"type":"interpolated","knots":{0:.5,8:.75,63:.5}}
 linear=expand_schedule(spec,64,"linear");nearest=expand_schedule(spec,64,"nearest")
 assert len(linear)==len(nearest)==64 and linear[0]==nearest[0]==.5 and linear[-1]==nearest[-1]==.5

def test_measured_only_defaults_unmeasured_layers_to_dense():
 values=expand_schedule({"type":"measured_only","default_retention":1.,"layers":{0:.5,8:.75}},64)
 assert values[0]==.5 and values[8]==.75 and values[7]==1.

def test_active_parameter_accounting():
 result=active_parameter_accounting([.5,1.],[100,100],1000)
 assert result["selected_ffn_parameters"]==150 and result["skipped_ffn_parameters"]==50 and result["estimated_active_model_parameters"]==950

