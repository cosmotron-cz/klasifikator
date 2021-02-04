/**
 * This class is the main view for the application. It is specified in app.js as the
 * "mainView" property. That setting automatically applies the "viewport"
 * plugin causing this view to become the body element (i.e., the viewport).
 *
 * TODO - Replace this content of this view to suite the needs of your application.
 */
Ext.define('ClassificationApp.view.main.Main', {
    extend: 'Ext.tab.Panel',
    xtype: 'app-main',

    requires: [
        'Ext.plugin.Viewport',
        'Ext.window.MessageBox',

        'ClassificationApp.view.main.MainController',
        'ClassificationApp.view.main.MainModel',
    ],

    controller: 'main',
    viewModel: 'main',

    ui: 'navigation',

    tabBarHeaderPosition: 1,
    titleRotation: 0,
    tabRotation: 0,

    header: {
        layout: {
            align: 'stretchmax'
        },
        title: {
            bind: {
                text: '{name}'
            },
            flex: 0
        },
        iconCls: 'fa-th-list'
    },

    tabBar: {
        flex: 1,
        layout: {
            align: 'stretch',
            overflowHandler: 'none'
        }
    },

    responsiveConfig: {
        tall: {
            headerPosition: 'top'
        },
        wide: {
            headerPosition: 'left'
        }
    },

    defaults: {
        bodyPadding: 20,
        tabConfig: {
            responsiveConfig: {
                wide: {
                    iconAlign: 'left',
                    textAlign: 'left'
                },
                tall: {
                    iconAlign: 'top',
                    textAlign: 'center',
                    width: 120
                }
            }
        }
    },

    items: [{
            title: 'Klasifikace',
            iconCls: 'fa-list-alt',
            items: [{
                xtype: 'classify'
            }],
            tabConfig: {
                listeners: {
                    click: function() {
                        ClassificationApp.getApplication().getStore('ClassificationData').load();
                        ClassificationApp.getApplication().getStore('Models').load();
                        ClassificationApp.getApplication().getStore('PlannedClassifications').load();
                    }
                }
            }
        }, 
        {
            title: 'Trénování',
            iconCls: 'fa-list-alt',
            items: [{
                xtype: 'train'
            }],
            tabConfig: {
                listeners: {
                    click: function() {
                        ClassificationApp.getApplication().getStore('TrainingData').load();
                        ClassificationApp.getApplication().getStore('PlannedTrainings').load();
                    }
                }
            }
        },
    ],
});
