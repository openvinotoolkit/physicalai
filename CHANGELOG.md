# Changelog

## [0.2.0](https://github.com/openvinotoolkit/physicalai/compare/v0.1.1...v0.2.0) (2026-09-21)


### ✨ Features

* add `physicalai-lerobot-plugin` ([#265](https://github.com/openvinotoolkit/physicalai/issues/265)) ([4ec496b](https://github.com/openvinotoolkit/physicalai/commit/4ec496b6067c101b8ac50beca65a25a89b6c4a9a))
* add agent skills foundation for runtime workflows ([#181](https://github.com/openvinotoolkit/physicalai/issues/181)) ([896c358](https://github.com/openvinotoolkit/physicalai/commit/896c358eda1e1b819e8ed36f1a4db4ec92868a36))
* add first party robot plugins ([#263](https://github.com/openvinotoolkit/physicalai/issues/263)) ([41e3ecf](https://github.com/openvinotoolkit/physicalai/commit/41e3ecf5e8d7a1c91d14234fceb0d040e2c7e07b))
* add from_config() and usage example ([#156](https://github.com/openvinotoolkit/physicalai/issues/156)) ([f479e9e](https://github.com/openvinotoolkit/physicalai/commit/f479e9eda6df39bdedd6c4606d8344de39c873f2))
* add IP camera support ([#230](https://github.com/openvinotoolkit/physicalai/issues/230)) ([9a56188](https://github.com/openvinotoolkit/physicalai/commit/9a56188c23cd51c0a24c756db3ac5de2e2538783))
* add policy source reset ([#237](https://github.com/openvinotoolkit/physicalai/issues/237)) ([64c48bb](https://github.com/openvinotoolkit/physicalai/commit/64c48bb3fb52ba367567c3889aa0516ecab5f857))
* add runtime.stop() ([#226](https://github.com/openvinotoolkit/physicalai/issues/226)) ([cb934cd](https://github.com/openvinotoolkit/physicalai/commit/cb934cd2e3b2c47236e8839444cdbb71cf7a7e83))
* **capture:** export config and cut over SharedCamera transport ([#205](https://github.com/openvinotoolkit/physicalai/issues/205)) ([04c41ef](https://github.com/openvinotoolkit/physicalai/commit/04c41ef46082dc008a456cf2025c35bf188e55ca))
* **ci:** add `release-please` ([#170](https://github.com/openvinotoolkit/physicalai/issues/170)) ([9937915](https://github.com/openvinotoolkit/physicalai/commit/993791553e4e5db999c5161daa12b4fd8e23b5fc))
* **ci:** add fuzzing ([#210](https://github.com/openvinotoolkit/physicalai/issues/210)) ([fde3e86](https://github.com/openvinotoolkit/physicalai/commit/fde3e86b4ea868d5e2659afb3cfc8d7b5dd989cf))
* **config:** add core component config engine ([#202](https://github.com/openvinotoolkit/physicalai/issues/202)) ([df559c2](https://github.com/openvinotoolkit/physicalai/commit/df559c2338b2bb108dec063a3ff720f88c9db824))
* **config:** unify Config API and absorb Studio config module ([#212](https://github.com/openvinotoolkit/physicalai/issues/212)) ([823f8bc](https://github.com/openvinotoolkit/physicalai/commit/823f8bcad0e402267d890e81316b71a8f61b689f))
* **config:** use jsonargparse as construction engine ([#227](https://github.com/openvinotoolkit/physicalai/issues/227)) ([6d8add9](https://github.com/openvinotoolkit/physicalai/commit/6d8add9a19a83a2159a65d2fe67363fabdd7be10))
* generic robot runtime ([#179](https://github.com/openvinotoolkit/physicalai/issues/179)) ([0a3e3ac](https://github.com/openvinotoolkit/physicalai/commit/0a3e3ac89b5831e81710f076624513d28ed83911))
* generic runtime - follow up ([#187](https://github.com/openvinotoolkit/physicalai/issues/187)) ([720adda](https://github.com/openvinotoolkit/physicalai/commit/720adda6c6ae5c2eb190060f5ad0e95c75d3852b))
* **inference:** add ov_smoke integration test suite ([#207](https://github.com/openvinotoolkit/physicalai/issues/207)) ([5b0b6ea](https://github.com/openvinotoolkit/physicalai/commit/5b0b6ea9d15eef7bc3765a9bbddbb76688156151))
* **inference:** add RLDX-1 OpenVINO preprocessing pipeline and VTC window callback ([#257](https://github.com/openvinotoolkit/physicalai/issues/257)) ([e68aa45](https://github.com/openvinotoolkit/physicalai/commit/e68aa45740ecc72ec880498dece98bbb9a455529))
* integrate studio plugin package ([#262](https://github.com/openvinotoolkit/physicalai/issues/262)) ([d31aa33](https://github.com/openvinotoolkit/physicalai/commit/d31aa33afc50bef3be0dbb53df834db01c1fc23b))
* load InferenceModel from hf hub ([#169](https://github.com/openvinotoolkit/physicalai/issues/169)) ([bcc6212](https://github.com/openvinotoolkit/physicalai/commit/bcc62120fffb009cf7ef5923d870c3e27a500df4))
* MolmoAct2 processors ([#234](https://github.com/openvinotoolkit/physicalai/issues/234)) ([15945cc](https://github.com/openvinotoolkit/physicalai/commit/15945cca350350e47c3a71868c02bb5e44a2e40a))
* Replace camera fingerprint string by dictionary with multiple identity ([#242](https://github.com/openvinotoolkit/physicalai/issues/242)) ([036edc2](https://github.com/openvinotoolkit/physicalai/commit/036edc2cd5ead102115a8fc4144ee8f04849fe90))
* robot serve CLI command ([#199](https://github.com/openvinotoolkit/physicalai/issues/199)) ([d4af359](https://github.com/openvinotoolkit/physicalai/commit/d4af359eb63cfac258f44564eb44aae6f5333fbd))
* robot transport via zenoh ([#189](https://github.com/openvinotoolkit/physicalai/issues/189)) ([d88ac1a](https://github.com/openvinotoolkit/physicalai/commit/d88ac1a2c371e547398a557a654f8f9e12628cc1))
* **robot:** export config and cut over SharedRobot transport ([#204](https://github.com/openvinotoolkit/physicalai/issues/204)) ([c8d38d7](https://github.com/openvinotoolkit/physicalai/commit/c8d38d7f6e9e71e97f9a1408fe101dd842908ebb))
* **runtime:** wire component config through RobotRuntime and CLI ([#206](https://github.com/openvinotoolkit/physicalai/issues/206)) ([f2fc7eb](https://github.com/openvinotoolkit/physicalai/commit/f2fc7eb23b00d4d389ce4e09cfe3e26491addd66))
* targeted code quality dirs ([#236](https://github.com/openvinotoolkit/physicalai/issues/236)) ([c488c52](https://github.com/openvinotoolkit/physicalai/commit/c488c524cbb400b4838dc53c0c422c5320f42ea0))
* Upgrade pyrealsense2 ([#246](https://github.com/openvinotoolkit/physicalai/issues/246)) ([9eaf436](https://github.com/openvinotoolkit/physicalai/commit/9eaf436c5fe2a62c154035cc67afdf689cf0ac2f))
* XR0 pre-post processing ([#256](https://github.com/openvinotoolkit/physicalai/issues/256)) ([f579176](https://github.com/openvinotoolkit/physicalai/commit/f5791761183b1f577e95bad49cd76d31d49d8154))


### 🐛 Bug Fixes

* add image_key_reorder_map and num_cameras to ResizeSmolVLA ([#231](https://github.com/openvinotoolkit/physicalai/issues/231)) ([4229fc0](https://github.com/openvinotoolkit/physicalai/commit/4229fc0d27546a53369979c2605aa131df4464ff))
* add reorder parameters to SmolVLA preprocessing ([#233](https://github.com/openvinotoolkit/physicalai/issues/233)) ([2528309](https://github.com/openvinotoolkit/physicalai/commit/2528309e55ab6a5f7f71c6dc23b8baae21f80108))
* align input type / layouts support across resize preprocessing nodes ([#162](https://github.com/openvinotoolkit/physicalai/issues/162)) ([f3ea7e6](https://github.com/openvinotoolkit/physicalai/commit/f3ea7e607587e65f1b4499c9827fc85f09e901b5))
* allow HuggingFace Hub snapshot symlinks ([#194](https://github.com/openvinotoolkit/physicalai/issues/194)) ([08fcd96](https://github.com/openvinotoolkit/physicalai/commit/08fcd96389fa9e326946df6affc5c19c4c731633))
* **ci:** exclude Lob detector from secret scan workflow ([#232](https://github.com/openvinotoolkit/physicalai/issues/232)) ([acfaf75](https://github.com/openvinotoolkit/physicalai/commit/acfaf75cc8b207bda1f5189c1660725e97bdb4f1))
* **config:** honor strict=False in Config.from_dict ([#253](https://github.com/openvinotoolkit/physicalai/issues/253)) ([8e40217](https://github.com/openvinotoolkit/physicalai/commit/8e4021703ef43387a835c6647b993cecc069ca85))
* **deps:** update dependency transformers to &gt;=5.5.0,&lt;5.6.0 [security] ([#192](https://github.com/openvinotoolkit/physicalai/issues/192)) ([6decbf5](https://github.com/openvinotoolkit/physicalai/commit/6decbf5058ea11ac8de40af5ac606e3002c56c66))
* **deps:** update transformers to &gt;=5.15.1,&lt;5.16.0 ([#267](https://github.com/openvinotoolkit/physicalai/issues/267)) ([8ae29a7](https://github.com/openvinotoolkit/physicalai/commit/8ae29a7477c9d31e97382a00971afcb639271373))
* dotted paths with an empty segment ([#223](https://github.com/openvinotoolkit/physicalai/issues/223)) ([e19a2e5](https://github.com/openvinotoolkit/physicalai/commit/e19a2e522dbe89e6319fa8de5a6896544badac18))
* enforce ValueError contract on invalid inputs; fix channels-last detection and pixel clamping in resize preprocessors ([#217](https://github.com/openvinotoolkit/physicalai/issues/217)) ([3683176](https://github.com/openvinotoolkit/physicalai/commit/3683176a049e00a3ed3874b1c6e8ed78dbabe1b1))
* harden artifact path containment and input key collision detection ([#215](https://github.com/openvinotoolkit/physicalai/issues/215)) ([d691348](https://github.com/openvinotoolkit/physicalai/commit/d6913485d9b576d231ed0b9ffb97bfac6b4d0c50))
* incorrect InferenceModel inputs with one camera setup ([#167](https://github.com/openvinotoolkit/physicalai/issues/167)) ([5848691](https://github.com/openvinotoolkit/physicalai/commit/58486917bddec243a79446d7332ca0ff7063afe1))
* **inference:** load callbacks from policy manifests ([#273](https://github.com/openvinotoolkit/physicalai/issues/273)) ([c5f6b71](https://github.com/openvinotoolkit/physicalai/commit/c5f6b71054d12776873de8ce7f8f0b309214b193))
* **inference:** support concrete InferenceFeature jsonargparse parsing ([#243](https://github.com/openvinotoolkit/physicalai/issues/243)) ([5215b84](https://github.com/openvinotoolkit/physicalai/commit/5215b843fa6440b129688d331f3386342a3847a4))
* physicalai studio plugin distribution ([#270](https://github.com/openvinotoolkit/physicalai/issues/270)) ([9455de6](https://github.com/openvinotoolkit/physicalai/commit/9455de69a3606af6038f65604bc8970778672a5b))
* reject non-finite stats in StatsNormalizer to prevent silent output corruption ([#216](https://github.com/openvinotoolkit/physicalai/issues/216)) ([61debd3](https://github.com/openvinotoolkit/physicalai/commit/61debd356aaffcbddc3c04e2848e9a2166c5c272))
* release workflow ([#271](https://github.com/openvinotoolkit/physicalai/issues/271)) ([c0750a3](https://github.com/openvinotoolkit/physicalai/commit/c0750a3b0f8edc5145ad9d95d6103fd444e315a3))
* restore canonical Apache 2.0 LICENSE text for GitHub recognition ([#219](https://github.com/openvinotoolkit/physicalai/issues/219)) ([ac537a7](https://github.com/openvinotoolkit/physicalai/commit/ac537a7930fbff7e259347c487ba94277418107f))
* **rldx1:** add runtime image resizing with stage-3 geometry controls ([#282](https://github.com/openvinotoolkit/physicalai/issues/282)) ([de49e1f](https://github.com/openvinotoolkit/physicalai/commit/de49e1fda4e38e0a5342b0a59a4cad11a1ee8de0))
* Small type fix ([#247](https://github.com/openvinotoolkit/physicalai/issues/247)) ([a7ee74d](https://github.com/openvinotoolkit/physicalai/commit/a7ee74d0a65bd8f27dc31d831210c7a905e92309))
* torch inference ([#185](https://github.com/openvinotoolkit/physicalai/issues/185)) ([9881601](https://github.com/openvinotoolkit/physicalai/commit/9881601325f724a0f00b1bf43f134d7f34fdef55))
* **trossen:** align WidowX state with recorded policies ([#259](https://github.com/openvinotoolkit/physicalai/issues/259)) ([a5e2750](https://github.com/openvinotoolkit/physicalai/commit/a5e275065f7a241a808b182bdcc31b62f6fc0a60))
* twin camera identity race ([#211](https://github.com/openvinotoolkit/physicalai/issues/211)) ([3e163cf](https://github.com/openvinotoolkit/physicalai/commit/3e163cfebc5f0fe6d97a6e792b4b0a55614d9a44))
* use correct Apache 2.0 license text ([#221](https://github.com/openvinotoolkit/physicalai/issues/221)) ([1be4325](https://github.com/openvinotoolkit/physicalai/commit/1be4325d6ccd8aeab7e6b80b1274b2e0e269747d))


### ♻️ Code Refactoring

* rename execution episode_id to incarnation ([#238](https://github.com/openvinotoolkit/physicalai/issues/238)) ([9be95d8](https://github.com/openvinotoolkit/physicalai/commit/9be95d88d5b9f8342f6952d8ec288fb108fa416e))
* rename skills to physicalai-runtime-* ([#183](https://github.com/openvinotoolkit/physicalai/issues/183)) ([90daa75](https://github.com/openvinotoolkit/physicalai/commit/90daa75b77045eea1b4bbd12c833aeccfd305d63))
* simplify RTC pipeline ([#255](https://github.com/openvinotoolkit/physicalai/issues/255)) ([89a6faf](https://github.com/openvinotoolkit/physicalai/commit/89a6faff37e2be91ae2f4c34746ec6dbeadb9f1d))
* speedup int8 resize ([#172](https://github.com/openvinotoolkit/physicalai/issues/172)) ([d3ccc7c](https://github.com/openvinotoolkit/physicalai/commit/d3ccc7cc7e7b65febb542b4e0f8417fdc9b41184))


### 📚 Documentation

* fix PyPI inference demo with hosted GIF ([#287](https://github.com/openvinotoolkit/physicalai/issues/287)) ([fafda32](https://github.com/openvinotoolkit/physicalai/commit/fafda32b3714560346293de2c338b57b189984c7))
* streamline issue templates ([#279](https://github.com/openvinotoolkit/physicalai/issues/279)) ([473d436](https://github.com/openvinotoolkit/physicalai/commit/473d4367bbac80ec816ea25806a7a6bf30f3ebf9))
* update banner image to include OpenVINO ([#154](https://github.com/openvinotoolkit/physicalai/issues/154)) ([b795346](https://github.com/openvinotoolkit/physicalai/commit/b795346a37f68539a6c8e130424589b2f20f0e7e))
* update RTC execution defaults and add update docstring ([#155](https://github.com/openvinotoolkit/physicalai/issues/155)) ([e3f2828](https://github.com/openvinotoolkit/physicalai/commit/e3f28285e9e2647c50baf17ee0b33614cbe29e7e))


### 🔧 Chores

* add security/trust model and suppress some SAST findings ([#254](https://github.com/openvinotoolkit/physicalai/issues/254)) ([5ada732](https://github.com/openvinotoolkit/physicalai/commit/5ada732cbc02e15345fb22b0e908165ae62932e7))
* bump lower bound for OpenVINO dependency ([#229](https://github.com/openvinotoolkit/physicalai/issues/229)) ([08c5596](https://github.com/openvinotoolkit/physicalai/commit/08c55968a16d4fb16a055f2f17bbebf397a15df8))
* bump OV version ([#283](https://github.com/openvinotoolkit/physicalai/issues/283)) ([cdb6a02](https://github.com/openvinotoolkit/physicalai/commit/cdb6a02c3f082c6dca8b453886a1c816f2828517))
* **ci:** add `dependency-review-action` config ([#163](https://github.com/openvinotoolkit/physicalai/issues/163)) ([d4a8655](https://github.com/openvinotoolkit/physicalai/commit/d4a8655f0446cf676d125bdf4d5f1c795e23f497))
* **ci:** add skills scanners in CI ([#190](https://github.com/openvinotoolkit/physicalai/issues/190)) ([a27d824](https://github.com/openvinotoolkit/physicalai/commit/a27d824ebc9cf6c267119d6d74b2b54ec927155d))
* **ci:** optimize check-paths patterns ([#252](https://github.com/openvinotoolkit/physicalai/issues/252)) ([d2085ce](https://github.com/openvinotoolkit/physicalai/commit/d2085ce98c3a41d4d60d59a0a8b2ec9a5482820c))
* clarify Copilot instructions apply to code review ([#225](https://github.com/openvinotoolkit/physicalai/issues/225)) ([f5017fb](https://github.com/openvinotoolkit/physicalai/commit/f5017fb47af7900e1303575b3a4bff64815c77c2))
* CLI runtime logging ([#209](https://github.com/openvinotoolkit/physicalai/issues/209)) ([da65bb9](https://github.com/openvinotoolkit/physicalai/commit/da65bb9ed894e44d8fe1309b547620aff6812e61))
* **deps:** bump tornado from 6.5.6 to 6.5.7 ([#165](https://github.com/openvinotoolkit/physicalai/issues/165)) ([ab028df](https://github.com/openvinotoolkit/physicalai/commit/ab028df86569204c9edcfac7bb56d6b8090f6434))
* **deps:** lock file maintenance ([#157](https://github.com/openvinotoolkit/physicalai/issues/157)) ([d59a4ba](https://github.com/openvinotoolkit/physicalai/commit/d59a4bad8176f24f685405a96514ed5de4c39230))
* **deps:** lock file maintenance ([#168](https://github.com/openvinotoolkit/physicalai/issues/168)) ([128036b](https://github.com/openvinotoolkit/physicalai/commit/128036b3bff10cde40daadf8c053caedaf087e10))
* **deps:** lock file maintenance ([#173](https://github.com/openvinotoolkit/physicalai/issues/173)) ([5383bcf](https://github.com/openvinotoolkit/physicalai/commit/5383bcf2c74fdce8c9749194534160fce0b9f978))
* **deps:** lock file maintenance ([#182](https://github.com/openvinotoolkit/physicalai/issues/182)) ([64ecaa5](https://github.com/openvinotoolkit/physicalai/commit/64ecaa58373d00c18672b79fc0ab14e56169544e))
* **deps:** lock file maintenance ([#188](https://github.com/openvinotoolkit/physicalai/issues/188)) ([045f8d9](https://github.com/openvinotoolkit/physicalai/commit/045f8d9b3a17439fb867ebf54dd4c9f85f7d342d))
* **deps:** lock file maintenance ([#198](https://github.com/openvinotoolkit/physicalai/issues/198)) ([ea863da](https://github.com/openvinotoolkit/physicalai/commit/ea863da72044cba20e7190cebe6c5a2096463435))
* **deps:** lock file maintenance ([#201](https://github.com/openvinotoolkit/physicalai/issues/201)) ([0db8782](https://github.com/openvinotoolkit/physicalai/commit/0db8782641edd7dfc53a6c26083f405d428a4c95))
* **deps:** lock file maintenance ([#214](https://github.com/openvinotoolkit/physicalai/issues/214)) ([41c6346](https://github.com/openvinotoolkit/physicalai/commit/41c6346b3774d7e656b6e6c9e054ec3a7d20c009))
* **deps:** lock file maintenance ([#228](https://github.com/openvinotoolkit/physicalai/issues/228)) ([30f103f](https://github.com/openvinotoolkit/physicalai/commit/30f103f44f8e2226656044b79185e89f4663b281))
* **deps:** lock file maintenance ([#240](https://github.com/openvinotoolkit/physicalai/issues/240)) ([43a8754](https://github.com/openvinotoolkit/physicalai/commit/43a8754a44c7a62c4135e9c1c1a0addf2552bfaa))
* **deps:** lock file maintenance ([#258](https://github.com/openvinotoolkit/physicalai/issues/258)) ([227f7ad](https://github.com/openvinotoolkit/physicalai/commit/227f7ad527b7008d8112981e26ecdf198bbe9d1f))
* **deps:** lock file maintenance ([#284](https://github.com/openvinotoolkit/physicalai/issues/284)) ([2fafa82](https://github.com/openvinotoolkit/physicalai/commit/2fafa820bb81798da92c9ed84f23d8542bc2f3c0))
* **deps:** update github actions ([#161](https://github.com/openvinotoolkit/physicalai/issues/161)) ([b671f52](https://github.com/openvinotoolkit/physicalai/commit/b671f52f4c39b612916008f1cc9f66ae66dae763))
* **deps:** update github actions ([#178](https://github.com/openvinotoolkit/physicalai/issues/178)) ([918fd14](https://github.com/openvinotoolkit/physicalai/commit/918fd14a7ebd15261810003f7b155e42f9efb11b))
* **deps:** update github actions ([#195](https://github.com/openvinotoolkit/physicalai/issues/195)) ([b6821f2](https://github.com/openvinotoolkit/physicalai/commit/b6821f2d4e9add5c380d757379339879b0b47f20))
* **deps:** update github actions ([#213](https://github.com/openvinotoolkit/physicalai/issues/213)) ([c71a581](https://github.com/openvinotoolkit/physicalai/commit/c71a581d91fa3332cbb564a3a8687bbd2c41c91e))
* **deps:** update github actions ([#239](https://github.com/openvinotoolkit/physicalai/issues/239)) ([c39fb50](https://github.com/openvinotoolkit/physicalai/commit/c39fb50b6c5dcb605228efd5e5957f4731fb9b3d))
* **deps:** update github actions ([#248](https://github.com/openvinotoolkit/physicalai/issues/248)) ([fe9f2fd](https://github.com/openvinotoolkit/physicalai/commit/fe9f2fd3b3f0125cf25918dca2ba3a078786aa4f))
* **deps:** update github actions ([#274](https://github.com/openvinotoolkit/physicalai/issues/274)) ([f3e68ba](https://github.com/openvinotoolkit/physicalai/commit/f3e68baf809f96e9f39ea21c0200d617e97e4fa1))
* **docs:** add badges into Readme ([#224](https://github.com/openvinotoolkit/physicalai/issues/224)) ([c23ce3c](https://github.com/openvinotoolkit/physicalai/commit/c23ce3cb909b0dcb4363f9760617d0e316006fad))
* **docs:** update copilot instructions ([#281](https://github.com/openvinotoolkit/physicalai/issues/281)) ([a3f522f](https://github.com/openvinotoolkit/physicalai/commit/a3f522f9554034d2d35e97bd2b0f72feae8b3067))
* **fuzzing:** increase fuzzing duration ([#222](https://github.com/openvinotoolkit/physicalai/issues/222)) ([615c2a0](https://github.com/openvinotoolkit/physicalai/commit/615c2a0c4de66d3127f89c10687d148bcc47f7e6))
* **main:** release physicalai-bimanual-so101-plugin 0.5.0 ([#278](https://github.com/openvinotoolkit/physicalai/issues/278)) ([3746e6b](https://github.com/openvinotoolkit/physicalai/commit/3746e6baeaede30bcfe38681dafebf562eb612df))
* **main:** release physicalai-lerobot-plugin 0.4.0 ([#280](https://github.com/openvinotoolkit/physicalai/issues/280)) ([04f067a](https://github.com/openvinotoolkit/physicalai/commit/04f067af8cc9e79723c486d86e2c73a0bf16fff4))
* **main:** release physicalai-rebot-b601-plugin 0.7.0 ([#277](https://github.com/openvinotoolkit/physicalai/issues/277)) ([3808fe3](https://github.com/openvinotoolkit/physicalai/commit/3808fe3dbd3dbe0870b0372d624980719ffc411b))
* **main:** release physicalai-stararm-plugin 0.3.0 ([#276](https://github.com/openvinotoolkit/physicalai/issues/276)) ([cc7d11c](https://github.com/openvinotoolkit/physicalai/commit/cc7d11cdf7f7557c3443aa77f34d9a9d6696cbc5))
* **main:** release physicalai-studio-plugin 0.2.0 ([#268](https://github.com/openvinotoolkit/physicalai/issues/268)) ([ec35b14](https://github.com/openvinotoolkit/physicalai/commit/ec35b140d8c63a257f5731847e1bd8f6569a5756))
* **main:** release physicalai-studio-plugin 0.2.0 ([#272](https://github.com/openvinotoolkit/physicalai/issues/272)) ([1d44a4c](https://github.com/openvinotoolkit/physicalai/commit/1d44a4cc1e07541abe6f6fd2bb75f304c98f9474))

## [0.1.1](https://github.com/openvinotoolkit/physicalai/compare/v0.1.0...v0.1.1) (2026-06-02)

**Full Changelog**: [v0.1.0...v0.1.1](https://github.com/openvinotoolkit/physicalai/compare/v0.1.0...v0.1.1)

## [0.1.0](https://github.com/openvinotoolkit/physicalai/releases/tag/v0.1.0) (2026-05-29)

Initial release.
