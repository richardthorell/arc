import { describe, expect, it } from 'vitest';

import { buildAssetLibraryScopeNavigation } from './assetLibraryNavigation';
import { assetLibraryScopePresentation } from './assetLibraryScopePresentation';

describe('assetLibraryScopePresentation', () => {
  it('presents project and user scopes as writable from the shared scope contract', () => {
    const scopes = buildAssetLibraryScopeNavigation([], [{ scope: 'user' }]);
    const project = scopes.find((scope) => scope.id === 'project')!;
    const user = scopes.find((scope) => scope.id === 'user')!;

    expect(assetLibraryScopePresentation(project)).toMatchObject({
      access: 'writable',
      accessLabel: 'Writable',
      ariaLabel: 'Project, writable',
    });
    expect(assetLibraryScopePresentation(user).description).toContain('Configured library mount.');
  });

  it('presents built-in and organization scopes as read only', () => {
    const scopes = buildAssetLibraryScopeNavigation([], [{ scope: 'organization' }]);
    const builtin = scopes.find((scope) => scope.id === 'builtin')!;
    const organization = scopes.find((scope) => scope.id === 'organization')!;

    expect(assetLibraryScopePresentation(builtin)).toMatchObject({
      access: 'read-only',
      accessLabel: 'Read only',
      ariaLabel: 'Built-in, read only',
    });
    expect(assetLibraryScopePresentation(organization)).toMatchObject({
      access: 'read-only',
      accessLabel: 'Read only',
      ariaLabel: 'Organization, read only',
    });
  });

  it('does not infer access from mount presence', () => {
    const scopes = buildAssetLibraryScopeNavigation([], [{ scope: 'organization' }, { scope: 'user' }]);
    const organization = scopes.find((scope) => scope.id === 'organization')!;
    const user = scopes.find((scope) => scope.id === 'user')!;

    expect(organization.mounted).toBe(true);
    expect(user.mounted).toBe(true);
    expect(assetLibraryScopePresentation(organization).access).toBe('read-only');
    expect(assetLibraryScopePresentation(user).access).toBe('writable');
  });
});
