import {
  UiAgentApprovalCard,
  UiAgentAssetCard,
  UiAgentAssetChoiceCard,
  UiAgentDiffCard,
  UiAgentErrorCard,
  UiAgentTaskCard,
  UiAgentToolCard,
  UiAgentViewportCard,
  UiButton,
} from '../ui';
import { EditorReferenceProvider } from '../services/EditorReferenceContext';

import './UiLabAgentCards.css';

const preview = (label: string) =>
  `data:image/svg+xml,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120"><rect width="160" height="120" fill="#24282d"/><text x="80" y="64" fill="#b9c0c7" font-family="sans-serif" font-size="14" text-anchor="middle">${label}</text></svg>`,
  )}`;

const choiceReferenceController = {
  resolve: async (reference: { kind: 'entity' | 'asset' | 'scene'; id: string }) => ({
    reference,
    label: reference.id === 'rock-a' ? 'Granite Rock 03' : 'Cliff Rock Large',
    subtitle: 'Model',
    thumbnailUrl: preview(reference.id === 'rock-a' ? 'Granite Rock' : 'Cliff Rock'),
  }),
  activate: async () => undefined,
  focus: async () => undefined,
};


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

      <EditorReferenceProvider controller={choiceReferenceController}>
        <UiAgentAssetChoiceCard
          title="Choose a rock model"
          prompt="I found two suitable project assets. Pick one before I place it."
          options={[
            { uri: 'arc://asset/rock-a', label: 'Granite Rock 03', reason: 'Closest silhouette to the reference.' },
            { uri: 'arc://asset/rock-b', label: 'Cliff Rock Large', reason: 'Better for a larger foreground shape.' },
          ]}
          onChoose={() => undefined}
        />
      </EditorReferenceProvider>

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
