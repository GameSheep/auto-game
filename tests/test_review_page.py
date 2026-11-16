"""图片审查页测试 —— manifest API / 图片服务 / 页面加载。"""

import asyncio
import json
import time
from pathlib import Path

from autogame.config import AppConfig
from autogame.agent.client import AgentClient

from test_e2e import ServerFixture
from autogame.sim.scene import SimCapture, SimPillar, SimWorld


def test_review_manifest_and_image():
    """审查 API：batch manifest 结构 + 图片服务 + 鉴权。

    100 图批量素材（assets/batch）是本地生成物、不入库（体积大且可重跑），
    所以 CI 上没有它：此时只验证鉴权/页面/目录穿越，内容断言自动跳过。
    """
    has_batch = (Path("assets/batch").is_dir()
                 and Path("assets/batch_report.json").exists())
    srv = ServerFixture()
    srv.start()

    import httpx

    async def main():
        async with httpx.AsyncClient(timeout=10) as hc:
            # 未鉴权
            r = await hc.get(f"{srv.url}/api/review/manifest?src=batch")
            assert r.status_code == 401
            # 鉴权 manifest（batch_report.json 由此前批量验证生成）
            r = await hc.get(f"{srv.url}/api/review/manifest?src=batch",
                             headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200
            d = r.json()
            if has_batch:
                assert d["total"] >= 90, f"batch 至少 90 项，实际 {d['total']}"
                assert d["a_press"] == 0, "全部图片按 A 必须为 0"
                first = d["items"][0]
                assert first["image"].startswith("/review/img?src=batch&name=")
                # 图片服务
                r = await hc.get(f"{srv.url}{first['image']}"
                                 f"&token=e2e-test-token")
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("image/")
            else:
                assert d["total"] == 0, "无素材时应返回空清单而不是报错"
            # 目录穿越防护
            r = await hc.get(f"{srv.url}/review/img?src=batch"
                             f"&name=..%2F..%2Fconfig.yaml&token=e2e-test-token")
            assert r.status_code in (400, 404), "目录穿越必须被拒"
            # 页面加载
            r = await hc.get(f"{srv.url}/review")
            assert r.status_code == 200 and "操作步骤" in r.text

    asyncio.run(main())


def test_review_replay_manifest():
    """回放帧 manifest（用真实导出的 user_farm 帧，若存在）。"""
    from pathlib import Path

    fp = Path("assets/frames_replay/user_farm/manifest.json")
    if not fp.exists():
        import pytest
        pytest.skip("回放帧未导出")
    srv = ServerFixture()
    srv.start()

    import httpx

    async def main():
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(
                f"{srv.url}/api/review/manifest?src=replay&name=user_farm",
                headers={"X-Token": "e2e-test-token"})
            assert r.status_code == 200
            d = r.json()
            assert d["total"] >= 150
            first = d["items"][0]
            assert first["image"].startswith("/review/img?src=replay&name=user_farm/")
            # 图片可达
            r = await hc.get(f"{srv.url}{first['image']}&token=e2e-test-token")
            assert r.status_code == 200

    asyncio.run(main())
