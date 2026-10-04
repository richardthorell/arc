import {
  UiAgentApprovalCard,
  UiAgentAssetCard,
  UiAgentDiffCard,
  UiAgentErrorCard,
  UiAgentTaskCard,
  UiAgentToolCard,
  UiAgentViewportCard,
  UiButton,
} from '../ui';

import './UiLabAgentCards.css';

export function UiLabAgentCards() {
  return (
    <div className="ui-lab-agent-card-gallery" aria-label="AI response card gallery">
      <UiAgentTaskCard
        details={
          <>
            <div>✓ Inspect cabin and current material</div>
            <div>✓ Compare reusable assets</div>
            <div>• Apply material polish</div>
            <div>• Verify in viewport</div>
          </>
        }
        metadata="2 of 4 steps"
        state="running"
        summary="Polishing the selected cabin while preserving its existing asset bindings."
        title="Polish cabin"
      />

      <UiAgentToolCard
        details={<pre>{'{\n  "sceneRevision": 42,\n  "entities": 18\n}'}</pre>}
        metadata="Step 1 · 84 ms"
        summary="Read the authoritative scene hierarchy before editing."
        title="scene.overview"
      />

      <UiAgentApprovalCard
        actions={
          <>
            <UiButton type="button" variant="ghost">
              Deny
            </UiButton>
            <UiButton type="button" variant="primary">
              Allow
            </UiButton>
          </>
        }
        state="pending"
        summary="ARC Built-in AI wants temporary edit access for one validated transaction."
        subtitle="Create cabin prop"
        title="Allow editor changes?"
      />

      <UiAgentDiffCard
        details={
          <>
            <div>Cabin.Transform.position.y: 0 → 0.15</div>
            <div>Cabin.MeshRenderer.material: Cabin_Old → Cabin_Polished</div>
            <div>FillLight.Light.intensity: 650 → 720</div>
          </>
        }
        metadata="3 changes · scene revision 43"
        summary="Three scene properties changed in the approved transaction."
        title="Scene changes"
      />

      <UiAgentViewportCard
        details="Capture includes color, depth, normals, and ObjectID channels."
        metadata="1280 × 720 · frame 1842"
        summary="Verification capture completed after the scene edit settled."
        title="Viewport capture"
      />

      <UiAgentAssetCard
        details="Project/Models/Architecture/SM_Cabin.glb\nStable asset: 41d9…c27a"
        metadata="Model · Project scope"
        summary="Reused the existing cabin model instead of rebuilding it from primitives."
        title="SM_Cabin"
      />

      <UiAgentErrorCard
        actions={
          <UiButton type="button" variant="ghost">
            Retry
          </UiButton>
        }
        defaultExpanded
        details="revision_conflict: scene revision changed from 43 to 44 before edit.apply."
        metadata="Retryable"
        summary="The material edit could not be applied because the scene changed."
        title="Material update failed"
      />
    </div>
  );
}
