# 多会话测试：完整选取清单

对应[数据说明](README.md)中的 W32/C32 用例，共 **100 个会话、80 个任务、65 个仓库、3379 次请求**。

原始数据集：[SWE-rebench OpenHands Trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)。本次保存的数据版本为 `35455389ab51bf5e2306bfd436ef72d0f98bf882`。

下表按已有会话清单排列，由原始 profiles 文件（本目录`inputs/baseline/causal_w32_noprimer.seed-20260725.profiles.jsonl`）逐项整理：

- **会话 ID** 对应原始数据的 `trajectory_id`，用于定位一次具体尝试；任务 ID 对应 `instance_id`，同一任务可出现多次。
- **所选原始轮次** 从 1 开始，首尾均包含。例如 44～87 表示选中 44 次模型调用；原始文件中的 `request_index` 从 0 开始，对应 43～86。
- **输入 token 数** 为所选首轮和末轮的完整历史长度，包含模板，不含会话标识符。
- 表中序号用于查阅会话；3379 次请求的实际发送次序由order 文件（同目录、同名`.order.jsonl`）记录。

| 序号 | 会话 ID（trajectory_id） | 任务 ID（instance_id） | 所选原始轮次 | 请求数 | 输入 token 数（首轮 → 末轮） |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `chatcmpl-64fff9fa637c8f0803edc89976795f88` | `jackdewinter__pymarkdown-259` | 44～87 | 44 | 64,047 → 107,645 |
| 2 | `chatcmpl-60769659499ed5e1f04a7f241cb3f454` | `troycomi__reportseff-65` | 45～93 | 49 | 64,081 → 117,438 |
| 3 | `chatcmpl-3db59ad16b1c94284ff824028e7b43d9` | `pybamm-team__PyBaMM-794` | 54～85 | 32 | 64,618 → 102,018 |
| 4 | `chatcmpl-72f50adb5020c05dbb7f40fbecca84a0` | `wemake-services__wemake-python-styleguide-1962` | 54～82 | 29 | 64,139 → 102,897 |
| 5 | `chatcmpl-5dc8915648372b5baefe31e36285dcc3` | `jupyter__nbgrader-1071` | 46～69 | 24 | 65,814 → 115,180 |
| 6 | `chatcmpl-410eb992c666c75e789b253b873ae089` | `keras-team__keras-nlp-131` | 52～82 | 31 | 64,730 → 117,312 |
| 7 | `chatcmpl-b72576eb1c7bf8d0aac39e351463149e` | `missionpinball__mpf-1137` | 69～80 | 12 | 64,407 → 110,926 |
| 8 | `chatcmpl-5413cabba72f25ee32032251f7f80b5e` | `pythological__etuples-7` | 80～100 | 21 | 67,302 → 108,569 |
| 9 | `chatcmpl-91b5debc4b1f34b74ebf0fb6a4da7669` | `jackdewinter__pymarkdown-259` | 52～82 | 31 | 64,430 → 120,179 |
| 10 | `chatcmpl-55c8428fd510e3197faefebe3d218af4` | `keras-team__keras-cv-1596` | 42～54 | 13 | 69,581 → 100,378 |
| 11 | `chatcmpl-d82b7d82f3ff822da952eaf652b59856` | `radiocosmology__alpenhorn-187` | 85～100 | 16 | 66,941 → 101,104 |
| 12 | `chatcmpl-8e197644b6c61c4553effc64fdb5db3e` | `pypest__pyemu-542` | 42～68 | 27 | 64,019 → 104,830 |
| 13 | `chatcmpl-21fd741b732af0e08300915f82105436` | `ORNL-MDF__Myna-22` | 67～100 | 34 | 64,277 → 111,873 |
| 14 | `chatcmpl-8303c113416fcb7b05bb13bf92a9a8bb` | `jupyter__nbgrader-1071` | 50～87 | 38 | 64,182 → 120,026 |
| 15 | `chatcmpl-5693fef6ac2b2f6c76402df77c2fa584` | `jackdewinter__pymarkdown-259` | 54～87 | 34 | 64,183 → 119,656 |
| 16 | `chatcmpl-756195ec2ea3fbdff3cba250c6bd8f21` | `pypa__hatch-1148` | 60～94 | 35 | 65,652 → 113,278 |
| 17 | `chatcmpl-fe813ff6116936ad5bd67df579669272` | `streamlink__streamlink-6381` | 50～79 | 30 | 67,381 → 109,310 |
| 18 | `chatcmpl-a625f5de265c2d5bb58b3f6f216b81af` | `nipy__nipype-2502` | 55～73 | 19 | 64,071 → 111,697 |
| 19 | `chatcmpl-3855c90b77145b62fc7a13fa9aafe813` | `scrapy__scrapy-2612` | 51～89 | 39 | 64,015 → 100,386 |
| 20 | `chatcmpl-94074b92fd0f9b979a8fee9c262b55fd` | `passaH2O__dorado-25` | 57～100 | 44 | 64,745 → 116,253 |
| 21 | `chatcmpl-5618cb5893c44976c24ab5678056bd42` | `encode__httpx-649` | 64～93 | 30 | 65,062 → 109,904 |
| 22 | `chatcmpl-dc0d318f828e6f2f85d11923078da888` | `nipy__nipype-2527` | 90～94 | 5 | 73,991 → 103,865 |
| 23 | `chatcmpl-9a79df50e250486c703cac06090bff33` | `PrefectHQ__prefect-3920` | 55～100 | 46 | 67,396 → 107,675 |
| 24 | `chatcmpl-8794bc090e429b80980daaf6460cca44` | `sqlfluff__sqlfluff-6123` | 68～100 | 33 | 70,269 → 107,103 |
| 25 | `chatcmpl-57e48388efd872df45dc1920e341949f` | `wemake-services__wemake-python-styleguide-3199` | 57～100 | 44 | 64,097 → 107,424 |
| 26 | `chatcmpl-f1baec1cbd83063a37d4f7da989e866c` | `jackdewinter__pymarkdown-269` | 56～82 | 27 | 64,481 → 121,887 |
| 27 | `chatcmpl-4a64c18857d566f1edc53ded68c9ab49` | `peterbe__hashin-125` | 45～100 | 56 | 64,239 → 102,632 |
| 28 | `chatcmpl-7bf43abb17be6372b1ebdcd901f1f050` | `sarugaku__requirementslib-201` | 57～100 | 44 | 70,486 → 111,771 |
| 29 | `chatcmpl-480d10d5367701162acc7301198cd2a9` | `Duke-GCB__DukeDSClient-283` | 66～100 | 35 | 64,170 → 108,260 |
| 30 | `chatcmpl-dbdd1feb98dddf2e2532d4cd8a295777` | `spacetelescope__jwst-3629` | 37～63 | 27 | 64,916 → 115,601 |
| 31 | `chatcmpl-aaad258e59bda017772b6b6789d52af3` | `juanifioren__django-oidc-provider-329` | 36～65 | 30 | 64,082 → 119,541 |
| 32 | `chatcmpl-749e6543debea505a013c98759323478` | `jackdewinter__pymarkdown-1195` | 48～76 | 29 | 68,117 → 121,387 |
| 33 | `chatcmpl-ffc5fd529294519d5d39ebb37e1c512e` | `alchemistry__alchemlyb-275` | 45～95 | 51 | 64,262 → 116,973 |
| 34 | `chatcmpl-bbfb811d2a2d8e1c85f5c427982d7244` | `whatshap__whatshap-412` | 55～100 | 46 | 64,448 → 103,310 |
| 35 | `chatcmpl-a0405bf952db8f2fb3c4a4785e0065f4` | `openradar__xradar-269` | 37～94 | 58 | 68,206 → 107,409 |
| 36 | `chatcmpl-874a27e90806e277a44d0a0d0da87882` | `jackdewinter__pymarkdown-259` | 61～100 | 40 | 66,193 → 100,399 |
| 37 | `chatcmpl-57b7b973f32408264bab0d9828197be7` | `postlund__pyatv-1004` | 49～65 | 17 | 67,528 → 107,208 |
| 38 | `chatcmpl-c9dcbdb8f9c5512cd45cc258202de00a` | `spacetelescope__jwst-3629` | 37～58 | 22 | 72,714 → 112,172 |
| 39 | `chatcmpl-ade218fe32485a16f429eca82cdbc32d` | `mmcdermott__MEDS_transforms-287` | 42～84 | 43 | 64,846 → 116,996 |
| 40 | `chatcmpl-45b538b4a906f4bd46a244e8f2d253ff` | `ml31415__numpy-groupies-48` | 56～99 | 44 | 65,502 → 116,728 |
| 41 | `chatcmpl-e783a948e04587bbcc5ac861462453d3` | `nipy__nipype-2527` | 58～66 | 9 | 70,795 → 105,055 |
| 42 | `chatcmpl-5000685606aed387ad4b07e5cd027631` | `pytest-dev__pytest-asyncio-1029` | 64～100 | 37 | 65,169 → 106,378 |
| 43 | `chatcmpl-564f12d3ce97793804f2203503d6c319` | `vyperlang__vyper-2627` | 63～82 | 20 | 65,660 → 105,781 |
| 44 | `chatcmpl-48d93e2353a4b12b7b1f06d427fe3e77` | `ejulio__spider-feeder-12` | 64～100 | 37 | 64,201 → 106,872 |
| 45 | `chatcmpl-1d18c79c71060a731d3e4b3f117cd8af` | `HBClab__NiBetaSeries-289` | 53～81 | 29 | 65,427 → 102,446 |
| 46 | `chatcmpl-807e8cbaa7d81331db70f02c81a06d8a` | `nipy__nipype-2527` | 76～88 | 13 | 70,210 → 102,926 |
| 47 | `chatcmpl-0e61dc5e5175c5a31292944854ad5dc0` | `numba__numba-5627` | 51～94 | 44 | 70,818 → 115,905 |
| 48 | `chatcmpl-4951b153fbcdbdb04a728604199b7bb5` | `mikedh__trimesh-609` | 59～100 | 42 | 64,497 → 100,898 |
| 49 | `chatcmpl-2c9eba6a72b0bc87390597b21682ee19` | `pymzml__pymzML-156` | 57～98 | 42 | 65,704 → 121,249 |
| 50 | `chatcmpl-94ca7834624d92c1d46510f3e24eaa90` | `juanifioren__django-oidc-provider-329` | 57～75 | 19 | 69,263 → 119,173 |
| 51 | `chatcmpl-6329abe1c4e243ddea2de7cf82a0aed9` | `neogeny__TatSu-35` | 60～100 | 41 | 65,405 → 105,702 |
| 52 | `chatcmpl-304218e9a0bd69c552ebe585cd38a8af` | `yt-dlp__yt-dlp-10531` | 36～93 | 58 | 67,269 → 114,953 |
| 53 | `chatcmpl-6023c26cb25a40ee95ed34dc0035c74b` | `optuna__optuna-3319` | 50～73 | 24 | 64,358 → 110,230 |
| 54 | `chatcmpl-39229df99456f05b6483d2ed0bd229b7` | `PennyLaneAI__pennylane-4251` | 49～100 | 52 | 64,017 → 115,879 |
| 55 | `chatcmpl-6d1e08d7721eb644858d216a126be4da` | `spacetelescope__jwst-3629` | 45～85 | 41 | 64,172 → 115,665 |
| 56 | `chatcmpl-90832ea5bd6d68b3ecc5f1c4c574025d` | `sqlfluff__sqlfluff-6123` | 81～100 | 20 | 64,187 → 100,659 |
| 57 | `chatcmpl-5b27bee8b7364a61f64db97e2787ccdf` | `tobymao__sqlglot-4415` | 41～100 | 60 | 64,439 → 100,323 |
| 58 | `chatcmpl-cb71bb364687707ab3e211d0d2bfcd7a` | `PlaidWeb__Publ-354` | 50～76 | 27 | 71,191 → 111,523 |
| 59 | `chatcmpl-a4e3a006cc634cff31084cc5cd0ddfd9` | `openradar__xradar-269` | 37～73 | 37 | 65,228 → 105,383 |
| 60 | `chatcmpl-a9c6827f65c2282221f72e150125578d` | `cdriehuys__django-simple-email-auth-26` | 49～71 | 23 | 64,153 → 107,951 |
| 61 | `chatcmpl-611b884975ca607444cce2bc1aedb310` | `mwouts__jupytext-588` | 45～100 | 56 | 68,101 → 103,764 |
| 62 | `chatcmpl-dea8fab7a6d5eeca0bab32a21af85352` | `canonical__charmcraft-915` | 57～97 | 41 | 64,517 → 104,464 |
| 63 | `chatcmpl-9b2bc6d52599398b6555ff76fb8299ea` | `nilearn__nilearn-5169` | 65～100 | 36 | 64,257 → 100,767 |
| 64 | `chatcmpl-b75ce6635af61e89d9cd0cbbc6269b72` | `vaidik__commentjson-46` | 72～100 | 29 | 65,412 → 111,028 |
| 65 | `chatcmpl-da292c28b7e33c434a7e5e009db4f1f8` | `troycomi__reportseff-60` | 43～100 | 58 | 65,712 → 111,479 |
| 66 | `chatcmpl-46a079878ed2153356476f5750906b31` | `nipy__nipype-2669` | 71～80 | 10 | 72,702 → 111,992 |
| 67 | `chatcmpl-6a5829159461105388b39e1f30993099` | `wemake-services__wemake-python-styleguide-2897` | 53～100 | 48 | 64,146 → 103,799 |
| 68 | `chatcmpl-6246fce33233e21b3c5ec802f05d90d1` | `nilearn__nilearn-5169` | 45～76 | 32 | 66,080 → 104,015 |
| 69 | `chatcmpl-1c5386145ec854c67dcbe9c6c76c395d` | `tox-dev__tox-2873` | 54～83 | 30 | 64,776 → 116,098 |
| 70 | `chatcmpl-4287537a253437f382fb78fde0f2c727` | `jackdewinter__pymarkdown-269` | 65～98 | 34 | 64,371 → 117,585 |
| 71 | `chatcmpl-b6060435eb823a2a70f29b001c3d57a7` | `jackdewinter__pymarkdown-1195` | 65～100 | 36 | 66,092 → 109,033 |
| 72 | `chatcmpl-fbdd853590776dcacb497f9582873641` | `juanifioren__django-oidc-provider-329` | 36～69 | 34 | 69,342 → 120,359 |
| 73 | `chatcmpl-d0e55cb74d7141663789659548602bd7` | `aesara-devs__aesara-986` | 67～100 | 34 | 64,232 → 101,697 |
| 74 | `chatcmpl-312e78f10e2333a0f641a795575a25f2` | `pdm-project__pdm-2910` | 41～70 | 30 | 66,538 → 117,124 |
| 75 | `chatcmpl-43e59486852b4792c5a5e5bc89bd042c` | `streamlink__streamlink-5565` | 33～83 | 51 | 67,309 → 100,357 |
| 76 | `chatcmpl-7856f07df8fd12aa24a64c9d0a746cff` | `reata__sqllineage-438` | 51～91 | 41 | 65,222 → 109,238 |
| 77 | `chatcmpl-d582dfcee48a30303a0395540437cf13` | `wemake-services__wemake-python-styleguide-1792` | 61～87 | 27 | 67,041 → 104,457 |
| 78 | `chatcmpl-ac3ede6bbd44ba57e120cdc46f10e852` | `keras-team__keras-cv-1882` | 60～98 | 39 | 64,074 → 116,394 |
| 79 | `chatcmpl-6865a6ea7aa2095e3fbe508591fbe804` | `kornia__kornia-1740` | 58～99 | 42 | 66,302 → 114,923 |
| 80 | `chatcmpl-29b4a451078876339e62040d322a9d23` | `pybamm-team__PyBaMM-797` | 30～73 | 44 | 64,053 → 108,853 |
| 81 | `chatcmpl-1a098389770e3b1efe210a59e127e38f` | `prjemian__spec2nexus-212` | 48～84 | 37 | 66,583 → 109,745 |
| 82 | `chatcmpl-b68981c193d0f72e14333896a9b07b87` | `pybamm-team__PyBaMM-797` | 53～79 | 27 | 64,046 → 100,600 |
| 83 | `chatcmpl-e1a45e742f829a462e8106d6d163a028` | `dask__dask-7734` | 52～88 | 37 | 64,109 → 116,381 |
| 84 | `chatcmpl-f684289ac7793989ea9defb117dd12b1` | `googleapis__python-spanner-122` | 73～99 | 27 | 69,022 → 104,815 |
| 85 | `chatcmpl-8deb5690918729d6cb1f2405a2a2b9eb` | `sqlfluff__sqlfluff-4974` | 61～100 | 40 | 64,616 → 112,236 |
| 86 | `chatcmpl-0e95e19d7a3078f7951aa6b309c65494` | `jlusiardi__tlv8_python-14` | 55～100 | 46 | 65,495 → 108,469 |
| 87 | `chatcmpl-91f718649e1aec7a616c6458d47df4e4` | `ESMValGroup__ESMValCore-2240` | 52～84 | 33 | 65,224 → 104,548 |
| 88 | `chatcmpl-8d88c9ac8107061fdd9e20e684ac5925` | `juanifioren__django-oidc-provider-329` | 36～57 | 22 | 66,473 → 117,470 |
| 89 | `chatcmpl-26acb46cdfda6111fb0c3c8755c37c52` | `nipy__nipype-2673` | 78～87 | 10 | 64,140 → 107,096 |
| 90 | `chatcmpl-f4a577d50f011ef3aad10f2ec11ddb0c` | `BAMWelDX__weldx-476` | 79～98 | 20 | 67,343 → 104,377 |
| 91 | `chatcmpl-920d67405cdd1d2a2a4c2854d890b644` | `twu__skjold-149` | 31～67 | 37 | 64,458 → 102,369 |
| 92 | `chatcmpl-0c4b4a0c7b1f81615285e2971dc3b290` | `jackdewinter__pymarkdown-259` | 49～72 | 24 | 66,898 → 109,498 |
| 93 | `chatcmpl-d2213615f6f24ddd360866c798c74e29` | `BoboTiG__ebook-reader-dict-2149` | 47～80 | 34 | 69,703 → 118,457 |
| 94 | `chatcmpl-7d10e9566a31efad62f4a3102c8bf1f8` | `pydicom__pydicom-1256` | 41～59 | 19 | 66,646 → 102,881 |
| 95 | `chatcmpl-dfacebb589a0e1aad9916a1f813b54d4` | `faucetsdn__chewie-109` | 54～82 | 29 | 64,529 → 101,661 |
| 96 | `chatcmpl-e168d8604c3cde57b205845e6b9e1071` | `yt-dlp__yt-dlp-12718` | 51～96 | 46 | 65,639 → 107,068 |
| 97 | `chatcmpl-c27b62698f7168823829da2cc3e07cf1` | `keras-team__keras-nlp-131` | 41～69 | 29 | 66,731 → 105,483 |
| 98 | `chatcmpl-8355be7aebc7c2d11b5262956776f0a3` | `conan-io__conan-5221` | 46～83 | 38 | 65,239 → 101,977 |
| 99 | `chatcmpl-b980f784646db76735a27f96ff9233c9` | `vyperlang__vyper-4385` | 61～95 | 35 | 67,612 → 106,962 |
| 100 | `chatcmpl-9284523187a903cfdcb3e571f6dccdfb` | `optuna__optuna-3319` | 37～69 | 33 | 64,680 → 110,576 |

**合计：100 个会话，3379 次请求。**
