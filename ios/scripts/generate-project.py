"""Generate an explicit deterministic Xcode source/resource manifest, no XcodeGen needed."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def ident(value): return hashlib.sha256(value.encode()).hexdigest()[:24].upper()
def quote(value): return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
objects = []
def obj(key, body):
    value = ident(key)
    objects.append(f'{value} = {{ {body} }};')
    return value
sources = sorted((root / 'Sources/VolleyCore').glob('*.swift')) + sorted((root / 'App').glob('*.swift')) + sorted((root / 'App').glob('*.mm'))
refs, builds = [], []
for path in sources:
    rel = path.relative_to(root).as_posix()
    ref = obj('ref:' + rel, f'isa = PBXFileReference; path = {quote(rel)}; sourceTree = "<group>"; lastKnownFileType = {"sourcecode.cpp.objcpp" if path.suffix == ".mm" else "sourcecode.swift"};')
    refs.append(ref)
    builds.append(obj('build:' + rel, f'isa = PBXBuildFile; fileRef = {ref};'))
resources = []
assets = root / 'App/Assets.xcassets'
if assets.exists():
    ref = obj('resource:assets', 'isa = PBXFileReference; path = "App/Assets.xcassets"; sourceTree = "<group>"; lastKnownFileType = folder.assetcatalog;')
    refs.append(ref)
    resources.append(obj('resource-build:assets', f'isa = PBXBuildFile; fileRef = {ref};'))
brand = obj('resource:volleysplice_logo.png', 'isa = PBXFileReference; path = "Fixtures/volleysplice_logo.png"; sourceTree = "<group>";')
refs.append(brand)
resources.append(obj('resource-build:volleysplice_logo.png', f'isa = PBXBuildFile; fileRef = {brand};'))
for name in ['model-1ca43e38eefc.json', 'model-9c92b8e9333f.json', 'suppression-overlap-exclusion-retrained.json', 'serving-side-85bc3325fbd4.json', 'side-switch-c2570481c30d.json', 'golden.json', 'base.bin', 'ios-editor-fixture.mp4', 'ios-export-fixture.mp4', 'overlay-fixture.mp4', 'overlay-fixture-90.mp4', 'overlay-fixture-180.mp4', 'overlay-fixture-270.mp4']:
    ref = obj('resource:' + name, f'isa = PBXFileReference; path = "Fixtures/{name}"; sourceTree = "<group>";')
    refs.append(ref)
    resources.append(obj('resource-build:' + name, f'isa = PBXBuildFile; fileRef = {ref};'))
framework = obj('opencv', 'isa = PBXFileReference; path = "../../wslmac-dependencies/opencv-4.12.0/opencv2.framework"; sourceTree = "<group>"; lastKnownFileType = wrapper.framework;')
refs.append(framework)
frameworkBuild = obj('opencv-build', f'isa = PBXBuildFile; fileRef = {framework};')
product = obj('product', 'isa = PBXFileReference; path = VolleySplice.app; sourceTree = BUILT_PRODUCTS_DIR; explicitFileType = wrapper.application;')
group = obj('group', f'isa = PBXGroup; children = ({",".join(refs + [product])}); sourceTree = "<group>";')
sourcePhase = obj('source-phase', f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(builds)}); runOnlyForDeploymentPostprocessing = 0;')
resourcePhase = obj('resource-phase', f'isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(resources)}); runOnlyForDeploymentPostprocessing = 0;')
frameworkPhase = obj('framework-phase', f'isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = ({frameworkBuild}); runOnlyForDeploymentPostprocessing = 0;')
configs = []
for name in ['Debug', 'Release']:
    settings = {
        'PRODUCT_NAME': 'VolleySplice', 'PRODUCT_BUNDLE_IDENTIFIER': 'com.vafrederico.VolleySplice',
        'DEVELOPMENT_TEAM': '<apple-development-team-id>', 'CODE_SIGN_STYLE': 'Automatic', 'CODE_SIGN_IDENTITY': 'Apple Development',
        'CURRENT_PROJECT_VERSION': '1', 'MARKETING_VERSION': '0.1.0', 'IPHONEOS_DEPLOYMENT_TARGET': '17.0',
        'SDKROOT': 'iphoneos', 'TARGETED_DEVICE_FAMILY': '1,2', 'SUPPORTED_PLATFORMS': 'iphoneos iphonesimulator',
        'ASSETCATALOG_COMPILER_APPICON_NAME': 'AppIcon',
        'SWIFT_VERSION': '5.0', 'SWIFT_OPTIMIZATION_LEVEL': '-O', 'ENABLE_TESTABILITY': 'YES' if name == 'Debug' else 'NO',
        'SWIFT_ACTIVE_COMPILATION_CONDITIONS': 'DEBUG' if name == 'Debug' else '',
        'CLANG_CXX_LANGUAGE_STANDARD': 'c++17', 'CLANG_ENABLE_MODULES': 'YES',
        'SWIFT_OBJC_BRIDGING_HEADER': 'App/OpenCVBridge.h',
        'FRAMEWORK_SEARCH_PATHS': '$(inherited) $(SRCROOT)/../../wslmac-dependencies/opencv-4.12.0',
        'OTHER_LDFLAGS': '-lc++ -framework Accelerate -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreGraphics -framework UIKit -framework Foundation -framework ImageIO -framework QuartzCore -framework VideoToolbox',
        'GENERATE_INFOPLIST_FILE': 'YES', 'INFOPLIST_FILE': 'App/Info.plist', 'INFOPLIST_KEY_CFBundleDisplayName': 'VolleySplice',
        'INFOPLIST_KEY_UILaunchScreen_Generation': 'YES', 'INFOPLIST_KEY_UIFileSharingEnabled': 'YES',
        'INFOPLIST_KEY_LSSupportsOpeningDocumentsInPlace': 'YES',
        # Device-specific orientations and full-screen compatibility live in
        # App/Info.plist, so project regeneration cannot override the iPad policy.
        'INFOPLIST_KEY_UIApplicationSceneManifest_Generation': 'YES',
    }
    configs.append(obj('config:' + name, 'isa = XCBuildConfiguration; name = '+name+'; buildSettings = {'+''.join(f'{k} = {quote(v)};' for k,v in settings.items())+'};'))
configList = obj('config-list', f'isa = XCConfigurationList; buildConfigurations = ({",".join(configs)}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target = obj('target', f'isa = PBXNativeTarget; name = VolleySplice; productName = VolleySplice; productType = "com.apple.product-type.application"; productReference = {product}; buildConfigurationList = {configList}; buildPhases = ({sourcePhase},{frameworkPhase},{resourcePhase}); dependencies = (); buildRules = ();')
test_targets = []
test_references = []
for test_name, folder, ui_test in [('VolleySpliceTests', 'Tests/AppIntegrationTests', False), ('VolleySpliceUITests', 'Tests/AppUITests', True)]:
    files = sorted((root / folder).glob('*.swift'))
    if not files: continue
    test_builds = []
    for file in files:
        relative = file.relative_to(root).as_posix()
        reference_id = obj('ref:' + relative, f'isa = PBXFileReference; path = {quote(relative)}; sourceTree = "<group>"; lastKnownFileType = sourcecode.swift;')
        test_builds.append(obj('build:' + relative, f'isa = PBXBuildFile; fileRef = {reference_id};'))
    test_product = obj(test_name + ':product', f'isa = PBXFileReference; path = {test_name}.xctest; sourceTree = BUILT_PRODUCTS_DIR; explicitFileType = wrapper.cfbundle;')
    phase = obj(test_name + ':sources', f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(test_builds)}); runOnlyForDeploymentPostprocessing = 0;')
    proxy = obj(test_name + ':proxy', f'isa = PBXContainerItemProxy; containerPortal = {ident("project")}; proxyType = 1; remoteGlobalIDString = {target}; remoteInfo = VolleySplice;')
    dependency = obj(test_name + ':dependency', f'isa = PBXTargetDependency; target = {target}; targetProxy = {proxy};')
    test_configs = []
    for config in ['Debug', 'Release']:
        values = {'PRODUCT_NAME': test_name, 'PRODUCT_BUNDLE_IDENTIFIER': 'com.vafrederico.' + test_name,
                  'SDKROOT': 'iphoneos', 'SUPPORTED_PLATFORMS': 'iphoneos iphonesimulator',
                  'TARGETED_DEVICE_FAMILY': '1,2', 'IPHONEOS_DEPLOYMENT_TARGET': '17.0',
                  'SWIFT_VERSION': '5.0', 'GENERATE_INFOPLIST_FILE': 'YES', 'CODE_SIGNING_ALLOWED': 'NO'}
        if ui_test: values['TEST_TARGET_NAME'] = 'VolleySplice'
        else:
            values['TEST_HOST'] = '$(BUILT_PRODUCTS_DIR)/VolleySplice.app/VolleySplice'
            values['BUNDLE_LOADER'] = '$(TEST_HOST)'
        test_configs.append(obj(test_name + ':' + config, 'isa = XCBuildConfiguration; name = ' + config + '; buildSettings = {' + ''.join(f'{k} = {quote(v)};' for k,v in values.items()) + '};'))
    config_list = obj(test_name + ':configs', f'isa = XCConfigurationList; buildConfigurations = ({",".join(test_configs)}); defaultConfigurationIsVisible = 0; defaultConfigurationName = Debug;')
    product_type = 'com.apple.product-type.bundle.ui-testing' if ui_test else 'com.apple.product-type.bundle.unit-test'
    test_target = obj(test_name + ':target', f'isa = PBXNativeTarget; name = {test_name}; productName = {test_name}; productType = "{product_type}"; productReference = {test_product}; buildConfigurationList = {config_list}; buildPhases = ({phase}); dependencies = ({dependency}); buildRules = ();')
    test_targets.append(test_target)
    test_references.append(f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{test_target}" BuildableName="{test_name}.xctest" BlueprintName="{test_name}" ReferencedContainer="container:VolleySplice.xcodeproj"/>')
project = obj('project', f'isa = PBXProject; mainGroup = {group}; buildConfigurationList = {configList}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; knownRegions = (en,Base); projectDirPath = ""; projectRoot = ""; targets = ({",".join([target] + test_targets)}); attributes = {{LastUpgradeCheck = 2630;}};')
path = root / 'VolleySplice.xcodeproj'
path.mkdir(exist_ok=True)
(path / 'project.pbxproj').write_text('// !$*UTF8*$!\n{archiveVersion = 1; classes = {}; objectVersion = 56; objects = {\n'+'\n'.join(objects)+f'\n}}; rootObject = {project};}}\n')
scheme = path / 'xcshareddata/xcschemes'
scheme.mkdir(parents=True, exist_ok=True)
reference = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="VolleySplice.app" BlueprintName="VolleySplice" ReferencedContainer="container:VolleySplice.xcodeproj"/>'
test_action = '<TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB"><Testables>' + ''.join('<TestableReference skipped="NO">' + ref + '</TestableReference>' for ref in test_references) + '</Testables></TestAction>'
(scheme / 'VolleySplice.xcscheme').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2630" version="1.3"><BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForRunning="YES" buildForTesting="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{reference}</BuildActionEntry></BuildActionEntries></BuildAction>{test_action}<LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></LaunchAction><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/></Scheme>''')
print(f'{len(sources)} sources in {path}')
