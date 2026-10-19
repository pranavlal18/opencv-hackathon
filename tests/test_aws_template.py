"""Static guards for the NOT-APPLIED SAM template (no AWS calls)."""
import json
import os

import pytest

yaml = pytest.importorskip("yaml")

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "..", "aws", "template.yaml")


def _load():
    with open(TPL) as f:
        return yaml.safe_load(f)


def test_template_parses():
    t = _load()
    assert t["Transform"] == "AWS::Serverless-2016-10-31"
    assert "DecideFunction" in t["Resources"]


def test_single_lambda_no_database_or_network():
    t = _load()
    kinds = {r: v["Type"] for r, v in t["Resources"].items()}
    # Exactly one compute resource + one stack-owned log group; nothing else.
    assert kinds == {"DecideFunction": "AWS::Serverless::Function",
                     "DecisionLogs": "AWS::Logs::LogGroup"}, kinds
    blob = open(TPL).read()
    for banned in ("AWS::DynamoDB::Table", "AWS::S3::Bucket",
                   "AWS::ApiGateway", "AWS::EC2::", "Bedrock", "SageMaker"):
        assert banned not in blob, banned


def test_log_group_retention_and_no_circular_dependency():
    t = _load()
    logs = t["Resources"]["DecisionLogs"]["Properties"]
    # 30-day bounded retention; name resolves via Ref to the generated
    # physical function name (/aws/lambda/<stack>-DecideFunction-<hash>).
    assert logs["RetentionInDays"] == 30
    assert logs["LogGroupName"] == {"Fn::Sub": "/aws/lambda/${DecideFunction}"}
    # Circular-proof: log group must depend ON the function (Ref only);
    # it must NOT point back with Fn::GetAtt (and the function resource
    # must not reference the log group back via LoggingConfig.LogGroup).
    assert "Fn::GetAtt" not in json.dumps(t["Resources"]["DecisionLogs"])
    fn = t["Resources"]["DecideFunction"]["Properties"]
    assert "LogGroup" not in fn.get("LoggingConfig", {})


def test_function_url_requires_iam():
    t = _load()
    fn = t["Resources"]["DecideFunction"]["Properties"]
    gl = t.get("Globals", {}).get("Function", {})
    # SAM FunctionUrlConfig enum is AWS_IAM (same IAM-auth semantics);
    # explicitly reject anonymous access.
    auth = fn["FunctionUrlConfig"]["AuthType"]
    assert auth == "AWS_IAM" and auth != "NONE"
    assert fn["Handler"] == "lambda_handler.handler"
    # Deliberately ABSENT (Task 10, user-approved): this account's regional
    # Lambda concurrency quota is 10 with a mandatory unreserved minimum of
    # 10, so Lambda rejects ANY per-function reservation (root cause of the
    # Task 9 deploy failure). The function shares the account's unreserved
    # Lambda concurrency pool; the binding concurrency cap is the
    # ACCOUNT-level quota (10), NOT a per-function cap of 5.
    assert "ReservedConcurrentExecutions" not in fn
    assert gl.get("Timeout", 99) <= 10 and gl.get("MemorySize", 999) <= 256
